# code.py

import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

# ============================
# Function Definitions
# ============================

def get_notes(file_path):
    """Extract notes and chords from a MIDI file."""
    notes = []
    try:
        midi = converter.parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return notes  # Return empty list if parsing fails

    print(f"Processing {file_path}")

    parts = instrument.partitionByInstrument(midi)
    if parts:  # If there are instrument parts
        for part in parts.parts:
            instr = part.getInstrument()
            print(f"Instrument found: {instr.instrumentName}")

            notes_to_parse = part.recurse()
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    else:
        # No instrument parts, attempt to parse flat notes
        notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def process_midi_files(directory):
    """Process all MIDI files in the given directory."""
    all_notes = []
    for file in glob.glob(f"{directory}/*.mid"):
        notes = get_notes(file)
        all_notes.extend(notes)
    return all_notes

# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Process jazz guitar MIDI files
    jazz_notes = process_midi_files('midi_files/jazz_guitar')

    # Process classical clarinet MIDI files
    classical_notes = process_midi_files('midi_files/classical_clarinet')

    # Combine notes from both genres
    all_notes = jazz_notes + classical_notes

    # Save notes to a file
    with open('data/notes.pkl', 'wb') as filepath:
        pickle.dump(all_notes, filepath)

    print(f"Total notes extracted: {len(all_notes)}")

    # ============================
    # 2. Prepare Sequences
    # ============================

    # Load notes
    with open('data/notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    print(f"Unique pitches: {n_vocab}")

    # Map pitches to integers and vice versa
    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    # Prepare input and output sequences
    sequence_length = 100
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    print(f"Total patterns: {n_patterns}")

    # Reshape and normalize the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    # One-hot encode the output
    network_output = to_categorical(network_output)

    # Optionally save the network_input and network_output for later use
    with open('data/network_input.pkl', 'wb') as filepath:
        pickle.dump(network_input, filepath)
    with open('data/network_output.pkl', 'wb') as filepath:
        pickle.dump(network_output, filepath)

    # ============================
    # 3. Define the Model
    # ============================

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print(model.summary())

    # ============================
    # 4. Train the Model
    # ============================

    # Define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.keras"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # Train the model
    model.fit(network_input, network_output, epochs=100, batch_size=64, callbacks=callbacks_list)

    # ============================
    # 5. Save the Final Model
    # ============================

    # Save the final model
    model.save('weights-improvement-final.keras')

    print("Model training complete and saved.")
