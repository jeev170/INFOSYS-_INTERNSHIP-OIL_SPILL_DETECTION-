# INFOSYS-INTERNSHIP-OIL_SPILL_DETECTION-

## Project Description
This project develops an AI-based system for oil spill detection and segmentation in satellite SAR images. It employs advanced deep learning segmentation techniques to generate pixel-wise oil spill maps, assisting in environmental monitoring and rapid intervention.

## Features

### Milestone 1: Data Collection, Exploration, and Preprocessing
- Loads and organizes satellite image data from Google Drive.
- Image resizing (256x256), normalization, and speckle noise reduction (SAR-specific).
- Data augmentation including flipping, rotating, brightness, and contrast adjustments. (Extra)
- Dataset statistics and visualization pre/post augmentation.
- Split into training, validation, and testing datasets.
- Save preprocessed images and masks as `.npy` files.

### Milestone 2: Model Development, Training, and Evaluation (Segmentation)
- Load preprocessed train, validation, and test datasets from Milestone 1.
- Build and train a U-Net deep learning model for pixel-wise oil spill segmentation.
- U-Net model includes batch normalization, dropout, and transpose convolution.
- Uses a combined Dice and Binary Cross-Entropy loss function for improved segmentation.
- Implements early stopping, learning rate reduction, and model checkpointing for robust training. (Extra)
- Real-time data augmentation during training to improve generalization. (Extra)
- Visualizes training loss and Dice coefficient progression.
- Evaluates performance on test set with Dice Coefficient, IoU, Precision, and Recall metrics.
- Saves prediction visualizations for qualitative inspection. (Extra)
