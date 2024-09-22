# AI-Powered Music Composer

An AI application that composes original music in the styles of jazz guitar and classical clarinet.

## Description

This project uses deep learning techniques to analyze musical patterns and generate new compositions. It leverages a Long Short-Term Memory (LSTM) neural network trained on MIDI files of jazz guitar and classical clarinet pieces. The generated music can be played back and downloaded through an interactive web application built with Streamlit.

## Features

- **Data Preprocessing**: Extracts notes and chords from MIDI files using `music21`.
- **Neural Network Model**: Utilizes an LSTM network built with TensorFlow and Keras.
- **Music Generation**: Generates new music sequences based on the trained model.
- **Audio Conversion**: Converts MIDI files to audio formats (WAV and MP3) using `FluidSynth` and `pydub`.
- **Interactive Interface**: Provides a user-friendly web app interface with Streamlit.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **pip package manager**

### Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Music-Composer.git
cd AI-Music-Composer
