# app.py

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from music21 import converter, instrument, note, chord, stream
import streamlit as st

# ============================
# Configuration Parameters
# ============================

MODEL_DIR = '../models/gpt2-music'
OUTPUT_DIR = 'outputs'
GENERATED_MIDI = os.path.join(OUTPUT_DIR, 'generated_output.mid')

# ============================
# Function Definitions
# ============================

def load_model():
    """Load the trained model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return tokenizer, model

def generate_music(tokenizer, model, prompt, max_length=SEQUENCE_LENGTH):
    """Generate music tokens based on the prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.encode("END")[0] if "END" in tokenizer.get_vocab() else None
        )
    generated_tokens = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_tokens

def tokens_to_midi(tokens, output_file=GENERATED_MIDI):
    """Convert generated tokens back to a MIDI file."""
    midi_stream = stream.Stream()
    tokens_list = tokens.split()
    current_tempo = 120  # Default tempo
    for token in tokens_list:
        if token.startswith("STYLE:"):
            continue  # Skip style tokens
        elif token.startswith("PITCH:") or token.startswith("CHORD:"):
            parts = token.split('_')
            if token.startswith("PITCH:"):
                pitch = parts[0].split(':')[1]
                volume = int(parts[1].split(':')[1])
                duration = float(parts[3].split(':')[1])
                n = note.Note(pitch)
                n.volume.velocity = volume
                n.duration.quarterLength = duration
                midi_stream.append(n)
            elif token.startswith("CHORD:"):
                pitches = parts[0].split(':')[1].split('.')
                volume = int(parts[1].split(':')[1])
                duration = float(parts[3].split(':')[1])
                c = chord.Chord(pitches)
                c.volume.velocity = volume
                c.duration.quarterLength = duration
                midi_stream.append(c)
        elif token.startswith("END"):
            break  # Stop generation
    midi_stream.write('midi', fp=output_file)
    return output_file

# ============================
# Streamlit App
# ============================

def main():
    st.title("🎵 AI Music Generator")
    st.write("Generate music by combining Classical mozart and jazz styles using a Transformer model.")

    tokenizer, model = load_model()

    with st.sidebar:
        st.header("Generation Settings")
        style = st.selectbox("Choose Style", options=["mozart", "jazz"])
        prompt = st.text_input("Enter a prompt (optional)", value=f"STYLE:{style}")
        generate_button = st.button("Generate Music")

    if generate_button:
        if not prompt.strip():
            st.warning("Please enter a valid prompt.")
        else:
            with st.spinner("Generating music..."):
                generated_tokens = generate_music(tokenizer, model, prompt)
                midi_file = tokens_to_midi(generated_tokens)
                st.success("Music generation complete!")
                st.audio(midi_file, format='audio/midi')
                st.write("Download the MIDI file:")
                with open(midi_file, 'rb') as f:
                    st.download_button('Download MIDI', f, file_name='generated_music.mid')

if __name__ == "__main__":
    main()
