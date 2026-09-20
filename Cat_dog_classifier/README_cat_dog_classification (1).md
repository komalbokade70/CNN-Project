# Cat vs Dog Image Classifier

A custom CNN built with TensorFlow/Keras to classify images as cat or dog, trained on the Microsoft Cats vs Dogs dataset from Kaggle. Built and trained in Google Colab (GPU runtime).

## Dataset
[Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) — ~25,000 images, 20,000 used for training and 4,998 for validation after an 80/20 split.

## How to run
1. Open cat_dog_classifier.ipynb in Google Colab.
2. Get your own kaggle.json from Kaggle → Account → Create New API Token.
3. Run all cells — the first cell prompts you to upload kaggle.json.

## Preprocessing
- Corrupted/truncated images removed with PIL's verify() before training.
- Images resized to 224x224, rescaled to [0,1].
- Data augmentation on the training set: shear, zoom, horizontal flip.

## Model architecture
4 Conv2D + BatchNorm + MaxPool blocks (32 → 64 → 128 → 256 filters), followed by a dense head (256 → 128 → 64) with dropout (0.30/0.25/0.25), sigmoid output for binary classification.

Trained with Adam (lr=0.0001), binary cross-entropy loss, and early stopping on validation loss (patience=3).

## Results
10 epochs, final training accuracy 90%, validation accuracy 87%.

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Cat   | 0.86      | 0.91   | 0.88     |
| Dog   | 0.90      | 0.85   | 0.88     |

*Overall accuracy: 88%* (4998 validation images)

## Files
- cat_dog_classifier.ipynb — full notebook: data prep, training, evaluation, and sample predictions
- requirements.txt — Python dependencies
- assets/ — saved plots used in this README (optional)

## Notes
- Paths are hardcoded to /content/... since this was built and run in Google Colab.
- The saved model (cat_dog_classifier.keras) is not included in this repo — rerun the notebook to regenerate it.
