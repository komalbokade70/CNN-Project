# Fashion MNIST Image Classification (CNN)
A convolutional neural network built with TensorFlow/Keras to classify grayscale images of clothing items into 10 categories, using the Fashion MNIST dataset.

## Dataset
- Source: Fashion MNIST (loaded directly via tensorflow.keras.datasets.fashion_mnist)
- 60,000 training images, 10,000 test images, each 28x28 grayscale
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
  
## Model Architecture
A custom CNN with three convolutional blocks followed by a dense classification head:
- 3x Conv2D → BatchNormalization → MaxPool2D blocks (32 → 64 → 128 filters, 3x3 kernels, ReLU)
- Flatten
- Dense(128, relu) → Dropout(0.25) → Dense(64, relu) → Dropout(0.25) → Dense(10, softmax)
#### Training setup:
- Optimizer: Adam (learning rate 0.001)
- Loss: Sparse categorical crossentropy
- Early stopping on validation loss (patience 3, restores best weights)
- 20% train/validation split, up to 10 epochs

## Result
| Metric | Value |
|--------|-------|
| Test Accuracy | 89.56% |
| Test Loss | 0.3114 |

Training and validation accuracy tracked closely through most of training with mild overfitting appearing in later epochs (train accuracy continuing to climb past ~92% while validation plateaus around 88–89%) — expected for a model this size on this dataset without heavier augmentation.

## Confusion Matrix
The model performs strongly across most classes (900+ correct out of 1,000 for Trouser, Sandal, Bag, Sneaker). The main weak spot is class 6 (Shirt), which is frequently confused with class 0 (T-shirt/top) and class 4 (Coat) — these three classes look visually similar at 28x28 grayscale resolution, and this confusion is a well-known, expected limitation of Fashion-MNIST rather than a bug in the model.

## What I'd improve next
- Data augmentation (small rotations/shifts) to reduce overfitting in later epochs
- A slightly deeper or wider network, or transfer learning from a pretrained backbone, to specifically target the Shirt/T-shirt/Coat confusion
- Learning rate scheduling instead of a fixed rate
