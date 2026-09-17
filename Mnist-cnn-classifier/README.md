# MNIST Digit Classification using CNN
- A convolutional neural network built with TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset, achieving 98.97% test accuracy.

## Dataset
- MNIST — 70,000 grayscale images of handwritten digits (28x28 pixels)
- 60,000 training images, 10,000 test images
- Loaded via
- tensorflow.keras.datasets.mnist

## Model Architecture
### Code
- Conv2D(32, 3x3, relu) → BatchNormalization → MaxPool2D(2x2)
- Conv2D(64, 3x3, relu) → BatchNormalization → MaxPool2D(2x2)
Flatten
- Dense(128, relu) → Dropout(0.3)
- Dense(10, softmax)

- Total params: 422,026 (421,834 trainable)
- Optimizer: Adam (learning rate = 0.001)
- Loss: Sparse Categorical Crossentropy
- Callback: EarlyStopping (monitor=val_loss, patience=3, restore_best_weights=True)

## Preprocessing
- Reshaped images to (28, 28, 1) for CNN input
- Normalized pixel values to [0, 1] by dividing by 255

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 98.97% |
| Test Loss | 0.0385 |

## Confusion Matrix
The model performs consistently well across all 10 digit classes, with most misclassifications occurring between visually similar digits (e.g., 4/9, 7/9).

## How to Run
Open the notebook in Google Colab or Jupyter
Install dependencies: pip install -r requirements.txt
Run all cells — the MNIST dataset downloads automatically via Keras

## Tech Stack
TensorFlow / Keras
NumPy
Matplotlib / Seaborn
Scikit-learn (confusion matrix, classification report)
