# app.py

import streamlit as st
import numpy as np
import os
import pickle
import random
import subprocess
import sys
import platform
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from pydub import AudioSegment
from io import BytesIO

# Set up paths
MODEL_PATH = 'weights-improvement-best.keras'  # Update with your model file
NOTES_PATH = 'data/notes.pkl'
SOUNDFONT_PATH = 'soundfonts/FluidR3_GM.sf2'  # Update with your .sf2 file name
OUTPUT_MIDI = 'streamlit_output.mid'
OUTPUT_WAV = 'streamlit_output.wav'
OUTPUT_MP3 = 'streamlit_output.mp3'

# Load necessary data
with open(NOTES_PATH, 'rb') as filepath:
    notes = pickle.load(filepath)

# Get all pitch names
pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)

# Map pitches to integers and vice versa
note_to_int = {note: number for number, note in enumerate(pitchnames)}
int_to_note = dict(enumerate(pitchnames))

# Prepare sequences used by the model
sequence_length = 100  # Use the same sequence length as in training

# Prepare input sequences
network_input = []
for i in range(0, len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])

# Reshape and normalize input
n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)

# Load the model
model = load_model(MODEL_PATH)

def generate_notes(model, network_input, pitchnames, n_vocab, generate_length=500):
    """Generate notes from the neural network based on a sequence of notes."""
    # Pick a random seed
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    int_to_note = {number: note for number, note in enumerate(pitchnames)}

    prediction_output = []

    # Generate notes
    for note_index in range(generate_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern[1:], [[index]], axis=0)

    return prediction_output

def create_midi(prediction_output, output_filename=OUTPUT_MIDI):
    """Convert the output from the prediction to notes and create a MIDI file from the notes."""
    offset = 0
    output_notes = []

    # Create note and chord objects
    for pattern in prediction_output:
        # Chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5  # Adjust as needed

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)

    # Convert MIDI to WAV
    midi_to_audio(output_filename, OUTPUT_WAV, SOUNDFONT_PATH)

    # Convert WAV to MP3
    sound = AudioSegment.from_wav(OUTPUT_WAV)
    sound.export(OUTPUT_MP3, format='mp3')

def midi_to_audio(midi_file, audio_file, soundfont_path):
    """Convert MIDI file to audio file using FluidSynth."""
    if platform.system() == 'Windows':
        subprocess.run([
            'fluidsynth',
            '-ni',
            soundfont_path,
            midi_file,
            '-F',
            audio_file,
            '-r', '44100'
        ], shell=True)
    else:
        subprocess.run([
            'fluidsynth',
            '-ni',
            soundfont_path,
            midi_file,
            '-F', audio_file,
            '-r', '44100'
        ])

# Streamlit App
st.title('AI-Powered Music Composer')

st.markdown("""
This application generates original music compositions using a neural network trained on jazz guitar and classical clarinet pieces.
""")

generate_length = st.slider('Select Composition Length (in notes)', 100, 1000, 500)

if st.button('Generate Music'):
    with st.spinner('Generating music...'):
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab, generate_length)
        create_midi(prediction_output)
    st.success('Music generated successfully!')

    # Play the audio file
    audio_file = open(OUTPUT_MP3, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

    # Provide download link
    st.download_button(label='Download MIDI File', data=open(OUTPUT_MIDI, 'rb'), file_name='generated_music.mid', mime='audio/midi')
    st.download_button(label='Download MP3 File', data=open(OUTPUT_MP3, 'rb'), file_name='generated_music.mp3', mime='audio/mp3')
