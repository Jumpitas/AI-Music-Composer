# **Music Composition with GPT-2**

This project aims to build a music generation model by training a GPT-2 language model on MIDI files. The model learns to generate musical sequences in various styles by encoding MIDI files into tokens and training on these tokenized sequences. The model can be fine-tuned to generate music in the styles of composers like Mozart or specific genres like jazz.

---

## **Table of Contents**
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Training the Model](#2-training-the-model)
  - [3. Generating Music](#3-generating-music)
- [Dataset](#dataset)
- [Customization](#customization)
- [Data Augmentation](#data-augmentation)
- [Future Improvements](#future-improvements)

---

## **Features**

- Converts MIDI files into tokenized sequences for music generation.
- Trains a GPT-2 model to generate music in different styles, e.g., Mozart or Jazz.
- Supports multiple MIDI files and styles for training.
- Implements data augmentation techniques to diversify training data.
- Uses Hugging Face’s `Trainer` class for efficient model training.
- Supports custom dataset loading and easy expansion to other music styles.

---

## **Project Structure**

```bash
├── data/
│   ├── midi_files/       # Directory for storing raw MIDI files by style
│   ├── processed/        # Directory for processed token data
├── models/
│   └── gpt2-music/       # Directory to save the trained model and tokenizer
├── outputs/              # Directory to save model outputs (generated music)
├── logs/                 # Directory for logging training progress
├── scripts/
│   └── music_training.py # Main script for data processing, training, and evaluation
├── README.md             # Project documentation
├── requirements.txt      # Required Python libraries
└── .venv/                # Virtual environment for the project
```

## **Prerequisites**

Ensure you have the following installed:

- **Python**: Version 3.8 or higher.
- **Virtual Environment**: `virtualenv` or `venv` for environment management.
- **MIDI Files**: A collection of MIDI files to train the model on.
- Basic understanding of Python and deep learning frameworks like **PyTorch**.

---

## **Installation**

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/music-composition-gpt2.git
    cd music-composition-gpt2
    ```

2. **Set up a virtual environment**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

---

## **Running the Project**

### **1. Data Preprocessing**

Before training, you need to preprocess the MIDI files into tokenized sequences.

- Place your MIDI files into the `data/midi_files/` directory.
- Organize them by subfolders representing different styles (e.g., `Mozart`, `Jazz`).
- Run the data processing script to generate tokens:

    ```bash
    python scripts/music_training.py
    ```

This script will:

- Parse MIDI files.
- Tokenize musical notes, chords, and other musical elements.
- Save the processed tokens in `data/processed/tokens.pkl`.

### **2. Training the Model**

Once the data is processed, you can train the GPT-2 model.

- Run the following command to start training:

    ```bash
    python scripts/music_training.py
    ```

Training will:

- Load the preprocessed tokens.
- Tokenize and prepare the dataset for GPT-2.
- Train the GPT-2 model on the music sequences.
- Save the trained model and tokenizer in the `models/gpt2-music/` directory.

### **3. Generating Music**

Once the model is trained, you can generate music by running the inference script (in development).

---

## **Dataset**

For better results, you can use large MIDI datasets such as:

- **Lakh MIDI Dataset (LMD)**: [Download](https://colinraffel.com/projects/lmd/)
- **MAESTRO Dataset**: [Download](https://magenta.tensorflow.org/datasets/maestro)
- **Kaggle MIDI Datasets**: [Find on Kaggle](https://www.kaggle.com/search?q=midi+dataset)

After downloading, place your new MIDI files into the `data/midi_files/` directory.

---

## **Customization**

### **Adding New Styles**

To train the model on new music styles:

1. Place your MIDI files into a subfolder under `data/midi_files/` (e.g., `data/midi_files/Beethoven/`).
2. Update the `STYLES` list in `music_training.py` to include the new style:

    ```python
    STYLES = ['Mozart', 'Jazz', 'Beethoven']
    ```

### **Adjust Model Parameters**

You can adjust the following parameters in `music_training.py`:

- **SEQUENCE_LENGTH**: Length of each input sequence.
- **BATCH_SIZE**: Batch size for training.
- **EPOCHS**: Number of epochs for training.
- **LEARNING_RATE**: Learning rate for the optimizer.

---

## **Data Augmentation**

To improve model performance and dataset diversity, you can apply data augmentation techniques:

- **Pitch Transposition**: Transpose all notes up or down by a certain number of semitones.
- **Tempo Changes**: Modify the speed of the MIDI files to create a larger variety of inputs.
- **Instrument Substitution**: Replace instruments in the MIDI files with others to diversify the data.

---

## **Future Improvements**

- **Inference Script**: Develop an interactive script for generating and exporting music as MIDI files.
- **Larger Dataset Support**: Improve the model's ability to handle large and complex datasets.
- **Web Interface**: Build a simple web interface to input themes and generate compositions dynamically.
- **Model Optimization**: Apply techniques like mixed-precision training or model pruning for faster training times.
- **Support for More Styles**: Expand the model to handle a broader range of music genres like rock, pop, classical symphonies, etc.

---

## **Contributing**

Feel free to contribute to this project by submitting issues or creating pull requests. All contributions are welcome!