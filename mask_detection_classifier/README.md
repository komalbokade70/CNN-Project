# Face Mask Detection (MobileNetV2)
Binary image classifier (with_mask / without_mask) built with transfer learning and fine-tuning on the
- Kaggle Face Mask Detection dataset.
  
## Pipeline
- 1. Crop faces from the Pascal VOC annotations (15% padding around each box)
- 2. Skip faces smaller than 24 px and drop the mask_weared_incorrect class
- 3. Split by source image into train / val / test (70 / 15 / 15), so no photo appears in two splits
- 4. MobileNetV2 (ImageNet weights) with preprocess_input, frozen base, custom head
- 5. Fine-tune the last 30 layers (learning rate 1e-5, BatchNorm frozen)
- 6. Keep the better stage by validation AUC, tune the decision threshold on validation
- 7. Report final metrics on the held-out test set
     
## Dataset after filtering
Of about 4,000 annotated faces, 2,235 were skipped as too small and 123 as mask_weared_incorrect.
The remaining 1,714 face crops are used:
|Split|with_mask|without_mask|
|---|---|
|Train|1026|166|
|Val|214|46|
|Test|228|34|
- The data is imbalanced (about 6 with_mask per 1 without_mask). Always predicting with_mask scores 87.0% on the test set, so accuracy alone is misleading.

## Results
- All splits scored the same way (clean images, dropout off):
|Split|Accuracy|ROC-AUC|
|---|---|
|Train|0.975|0.997|
|Validation|0.931|0.979|
|Test|0.950|0.985|
without_mask on the test set at threshold 0.5: precision 0.784, recall 0.853 (F1 about 0.82).

## Limitations
- Results apply to faces of at least 24 px. More than half of the faces in the dataset are smaller and were excluded, so these numbers are not comparable to a model evaluated on all faces.
- The test set is small (34 without_mask faces), so metrics can shift by several points with a different split.
- mask_weared_incorrect was excluded; a 3-class model is a natural extension.
- Train accuracy is higher than validation (gap of about 4 points), which indicates mild overfitting.

## Run
- Open face_mask_detection.ipynb in Google Colab
- Upload your kaggle.json when prompted (never commit it; it is in .gitignore)
- Run all cells. Figures are saved to assets/.
