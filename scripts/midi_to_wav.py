# midi_to_wav.py

import os
from midi2audio import FluidSynth

# ============================
# Configuration Parameters
# ============================

OUTPUT_DIR = 'outputs'
WAV_DIR = 'outputs/wav'
SOUND_FONT = 'soundfonts/FluidR3_GM.sf2'  # Path to your soundfont file

os.makedirs(WAV_DIR, exist_ok=True)

def midi_to_wav(midi_file, wav_file):
    """Convert a MIDI file to WAV format using FluidSynth."""
    fs = FluidSynth(sound_font=SOUND_FONT)
    fs.midi_to_audio(midi_file, wav_file)
    print(f"Converted {midi_file} to {wav_file}")

if __name__ == "__main__":
    for file in os.listdir(OUTPUT_DIR):
        if file.lower().endswith(('.mid', '.midi')):
            midi_path = os.path.join(OUTPUT_DIR, file)
            wav_filename = os.path.splitext(file)[0] + '.wav'
            wav_path = os.path.join(WAV_DIR, wav_filename)
            midi_to_wav(midi_path, wav_path)
