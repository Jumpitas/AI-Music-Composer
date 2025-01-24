# Music Composition with GPT-2

This project focuses on creating a music generation model by training a GPT-2 language model on MIDI files. The model is trained to compose music sequentially by encoding MIDI files into tokens and learning from these tokenized sequences. It can be fine-tuned to produce music in popular styles such as Mozart or jazz.

---

## Table of Contents
1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Running the Project](#running-the-project)  
    1. [Data Preprocessing](#1-data-preprocessing)  
    2. [Training the Model](#2-training-the-model)  
    3. [Generating Music](#3-generating-music)  
6. [Dataset](#dataset)  
7. [Customization](#customization)  
    1. [Adding New Styles](#adding-new-styles)  
    2. [Adjust Model Parameters](#adjust-model-parameters)  
8. [Data Augmentation](#data-augmentation)  
9. [Future Improvements](#future-improvements)  

---

## Features

- **Transforms MIDI files** into tokenized sequences for music generation.
- **Trains a GPT-2 model** to generate various styles of music (e.g., Mozart, Jazz).
- **Supports multiple MIDI files** for training, covering different music styles.
- **Data augmentation** techniques are implemented to generate a more diverse and well-distributed training dataset.
- **Efficient training** via Hugging Face’s `Trainer` class.
- **Flexible dataset sources**: Configure different types of music files and easily add new ones.

---

## Project Structure

```
├── data/
│   ├── midi_files/    # Directory for storing raw MIDI files by style
│   ├── processed/     # Directory for processed token data
├── models/
│   └── gpt2-music/    # Directory to save the trained model and tokenizer
├── outputs/           # Directory to save model outputs (generated music)
├── logs/              # Directory for logging training progress
├── scripts/
│   └── music_training.py  # Main script for data processing, training, and evaluation
├── README.md          # Project documentation
├── requirements.txt   # Required Python libraries
└── .venv/             # Virtual environment for the project
```

---

## Prerequisites

Make sure you have the following installed:

- **Python**: Version 3.8 or newer.
- **Virtual Environment**: `virtualenv` or `venv` for environment management.
- **MIDI Files**: A collection of MIDI files to train the model on.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Jumpitas/AI-Music-Composer.git
   cd AI-Music-Composer
   ```

2. **Set up a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   *On Windows use:* 
   ```bash
   .venv\Scripts\activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

### 1. Data Preprocessing

- **Place your MIDI files** into the `data/midi_files/` directory.  
- Organize your MIDI files into style-specific subfolders (e.g., `Mozart`, `Jazz`, etc.).
- **Run the data processing script** to convert the MIDI files into token sequences:

  ```bash
  python scripts/music_training.py
  ```

  This script will:
  - Parse musical notes from the MIDI files.
  - Tokenize music data into notes, chords, and other musical elements.
  - Save the processed tokens into `data/processed/tokens.pkl`.

### 2. Training the Model

After preprocessing the data, you can train the GPT-2 model:

```bash
python scripts/music_training.py
```

- This will:
  - Load the preprocessed tokens.
  - Tokenize and prepare the dataset for GPT-2.
  - Train the GPT-2 model on the music sequences.
  - Save the trained model and tokenizer in `models/gpt2-music/`.

### 3. Generating Music

- The model has a **sequence-to-sequence approach** for music generation.
- You can **provide a text prompt** describing the music you want, or follow the examples in the repository.

---

## Dataset

For **best results**, use large MIDI datasets such as:

- **Lakh MIDI Dataset (LMD)**: [Download](https://colinraffel.com/projects/lmd/)
- **MAESTRO Dataset**: [Download](https://magenta.tensorflow.org/datasets/maestro)
- **Kaggle MIDI Datasets**: [Find on Kaggle](https://www.kaggle.com/)

After downloading, place your new MIDI files in the `data/midi_files/` directory.

---

## Customization

### Adding New Styles

- Create a **new subfolder** under `data/midi_files/` for your desired style (e.g., `data/midi_files/Beethoven/`).
- **Add your MIDI files** to that folder.
- **Update the STYLES list** in `music_training.py` to include the new style:

  ```python
  STYLES = ['Mozart', 'Jazz', 'Beethoven']
  ```

### Adjust Model Parameters

Inside `music_training.py`, you can modify parameters to suit your needs:
- `SEQUENCE_LENGTH`: Length of each input sequence.
- `BATCH_SIZE`: Batch size for training.
- `EPOCHS`: Number of epochs for training.
- `LEARNING_RATE`: Learning rate for the optimizer.

---

## Data Augmentation

To enhance model performance and increase dataset variety, you can use:

- **Tempo Changes**: Speeding up or slowing down MIDI files to introduce variations.
- **Instrument Substitution**: Changing instruments within the MIDI files for more diverse data.

---

## Future Improvements

- **Advanced model architectures** (e.g., GPT-3, Music Transformer) for richer music generation.
- **Longer sequence handling** to capture extended musical compositions.
- **User-friendly interfaces** for interactive music generation.
- **Fine-grained control** over musical structure, instrumentation, and style.
