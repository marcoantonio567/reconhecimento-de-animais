# Reconhecimento de Animais

Animal image classification using a TensorFlow/Keras model and a label list.

## Project structure

- `reconhecimento-de-animais/`
  - `kereas.py`: classify a single image file
  - `opencv.py`: real-time webcam classification (press `ESC` to exit)
  - `graficos.py`: example plots (uses sample data)
  - `labels.txt`: class labels
  - `keras_model.h5`: trained model (file name may vary)

## Requirements

- Python 3
- A working webcam (only for `opencv.py`)

Install dependencies:

```bash
pip install -r reconhecimento-de-animais/requirements.txt
```

`graficos.py` uses extra libraries not listed in `requirements.txt`:

```bash
pip install seaborn matplotlib scikit-learn
```

## Labels / classes

The classes are defined in `reconhecimento-de-animais/labels.txt`:

- cachorro / dog
- cobra / snake
- tubarão / shark 
- baiacu / puffer fish
- porco / pig
- papagaio / parrot
- jacare / alligator
- pinguim / penguin
- salamandra /  salamander
- sapo / frog

## Usage

Run commands from the `reconhecimento-de-animais` folder so the scripts can find `labels.txt` and the model file using relative paths:

```bash
cd reconhecimento-de-animais
```

### 1) Classify a single image

Edit `kereas.py` and replace `<IMAGE_PATH>` with the path to your image, then run:

```bash
python kereas.py
```

### 2) Webcam classification

```bash
python opencv.py
```

Press `ESC` to close the window and stop the script.

### Model file name note

The scripts load `keras_Model.h5` by default. If your model file is named differently (for example `keras_model.h5`), rename it or update the `load_model(...)` path in the scripts.
