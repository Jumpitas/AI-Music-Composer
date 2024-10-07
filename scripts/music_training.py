# music_training.py

import os
import pickle
import logging
from music21 import converter, instrument, note, chord
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from tqdm import tqdm

# ============================
# Configuration Parameters
# ============================

DATA_DIR = 'data/midi_files'
PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models/gpt2-music'
OUTPUT_DIR = 'outputs'

SEQUENCE_LENGTH = 1024  # Transformer context length
BATCH_SIZE = 2  # Adjust based on GPU memory
EPOCHS = 10
LEARNING_RATE = 5e-5
STYLES = ['Mozart', 'Jazz']  # Styles to include

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


# ============================
# Function Definitions
# ============================

def get_notes(file_path, style):
    """
    Extract notes, chords, instruments, volumes, durations, and style from a MIDI file.

    Parameters:
    - file_path (str): Path to the MIDI file.
    - style (str): Musical style (e.g., 'Mozart', 'Jazz').

    Returns:
    - List[str]: List of encoded tokens.
    """
    notes = []
    try:
        midi = converter.parse(file_path)
    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return notes

    logging.info(f"Processing {file_path} as style: {style}")

    parts = instrument.partitionByInstrument(midi)
    if parts:  # File has instrument parts
        for part in parts.parts:
            instr = part.getInstrument()
            instr_name = instr.instrumentName if instr.instrumentName else "Unknown"
            notes_to_parse = part.recurse()
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    pitch = str(element.pitch)
                    volume = int(element.volume.velocity) if element.volume.velocity else 70  # Default volume
                    duration = element.duration.quarterLength
                    token = f"PITCH:{pitch}_V:{volume}_I:{instr_name}_D:{duration}"
                    notes.append(token)
                elif isinstance(element, chord.Chord):
                    pitches = '.'.join(str(n) for n in element.normalOrder)
                    volume = int(element.volume.velocity) if element.volume.velocity else 70
                    duration = element.duration.quarterLength
                    token = f"CHORD:{pitches}_V:{volume}_I:{instr_name}_D:{duration}"
                    notes.append(token)
    else:  # File has flat notes
        notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                pitch = str(element.pitch)
                volume = int(element.volume.velocity) if element.volume.velocity else 70
                duration = element.duration.quarterLength
                token = f"PITCH:{pitch}_V:{volume}_D:{duration}"
                notes.append(token)
            elif isinstance(element, chord.Chord):
                pitches = '.'.join(str(n) for n in element.normalOrder)
                volume = int(element.volume.velocity) if element.volume.velocity else 70
                duration = element.duration.quarterLength
                token = f"CHORD:{pitches}_V:{volume}_D:{duration}"
                notes.append(token)
    return notes


def process_style(style):
    """
    Process all MIDI files for a given style.

    Parameters:
    - style (str): Musical style (e.g., 'Mozart', 'Jazz').

    Returns:
    - List[str]: List of encoded tokens for the style.
    """
    all_notes = []
    style_dir = os.path.join(DATA_DIR, style)
    if not os.path.exists(style_dir):
        logging.warning(f"Style directory {style_dir} does not exist. Skipping.")
        return all_notes

    for root, _, files in os.walk(style_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                file_path = os.path.join(root, file)
                notes = get_notes(file_path, style)
                if notes:
                    all_notes.append(f"STYLE:{style}")
                    all_notes.extend(notes)
                    all_notes.append("END")  # Optional end token
    return all_notes


def process_midi_files():
    """
    Process all MIDI files and encode them into tokens with style information.

    Returns:
    - List[str]: Combined list of all tokens from all styles.
    """
    all_notes = []
    for style in STYLES:
        all_notes.extend(process_style(style))
    return all_notes


def save_tokens(notes, file_path):
    """
    Save tokens to a pickle file.

    Parameters:
    - notes (List[str]): List of tokens to save.
    - file_path (str): Path to the pickle file.
    """
    with open(file_path, 'wb') as fp:
        pickle.dump(notes, fp)
    logging.info(f"Saved tokens to {file_path}")


def load_tokens(file_path):
    """
    Load tokens from a pickle file.

    Parameters:
    - file_path (str): Path to the pickle file.

    Returns:
    - List[str]: Loaded list of tokens.
    """
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


def create_tokenizer(tokens, tokenizer_path='models/tokenizer'):
    """
    Create and save a GPT-2 tokenizer based on the unique tokens.

    Parameters:
    - tokens (List[str]): List of all tokens.
    - tokenizer_path (str): Path to save the tokenizer.

    Returns:
    - GPT2Tokenizer: The created tokenizer.
    """
    unique_tokens = sorted(set(tokens))
    # Initialize a new tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add new tokens
    tokenizer.add_tokens(unique_tokens)
    # Save the tokenizer
    tokenizer.save_pretrained(tokenizer_path)
    logging.info(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer


def encode_tokens(tokens, tokenizer):
    """
    Encode tokens using the tokenizer.

    Parameters:
    - tokens (List[str]): List of tokens to encode.
    - tokenizer (GPT2Tokenizer): The tokenizer to use.

    Returns:
    - List[int]: List of encoded token IDs.
    """
    # Join tokens into a single string separated by spaces
    text = ' '.join(tokens)
    encoded = tokenizer(text, return_tensors='pt', max_length=SEQUENCE_LENGTH, truncation=True)
    return encoded['input_ids'][0].tolist()


def prepare_dataset(encoded_tokens):
    """
    Prepare dataset for training by creating input sequences.

    Parameters:
    - encoded_tokens (List[int]): List of encoded token IDs.

    Returns:
    - List[List[int]]: List of input sequences.
    """
    sequences = []
    for i in tqdm(range(0, len(encoded_tokens) - SEQUENCE_LENGTH, SEQUENCE_LENGTH)):
        seq = encoded_tokens[i:i + SEQUENCE_LENGTH]
        sequences.append(seq)
    logging.info(f"Total sequences: {len(sequences)}")
    return sequences


# ============================
# Main Execution Block
# ============================

if __name__ == "__main__":
    # Step 1: Process MIDI files and extract tokens
    all_notes = process_midi_files()
    logging.info(f"Total tokens extracted: {len(all_notes)}")

    # Step 2: Save tokens
    tokens_file = os.path.join(PROCESSED_DIR, 'tokens.pkl')
    save_tokens(all_notes, tokens_file)

    # Step 3: Create and save tokenizer
    tokenizer = create_tokenizer(all_notes, tokenizer_path=os.path.join(MODEL_DIR, 'tokenizer'))

    # Step 4: Encode tokens
    encoded_tokens = encode_tokens(all_notes, tokenizer)

    # Step 5: Prepare dataset
    sequences = prepare_dataset(encoded_tokens)

    # Step 6: Create Hugging Face Dataset
    dataset = Dataset.from_dict({"input_ids": sequences})
    # Duplicate input_ids for labels
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]}, batched=False)
    # Split into train and validation
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    # Step 7: Initialize model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))  # Adjust token embeddings

    # Step 8: Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        save_total_limit=3,
    )

    # Step 9: Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Step 10: Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Step 11: Train the model
    trainer.train()

    # Step 12: Save the final model
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    logging.info(f"Model and tokenizer saved to {MODEL_DIR}")
