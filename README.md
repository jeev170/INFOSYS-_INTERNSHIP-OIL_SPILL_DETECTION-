# INFOSYS-INTERNSHIP-OIL_SPILL_DETECTION-

## Project Submissions

---

## Project Description

This project is an AI-powered system for automated oil spill detection and segmentation using satellite images. The goal is to help environmental monitoring agencies identify oil spill regions quickly and efficiently. The workflow covers specialized preprocessing for SAR imagery and a deep learning classification approach.

---

## Features

### Milestone 1: Data Collection, Exploration, & Preprocessing
- Loads satellite image data automatically from a Google Drive folder.
- Resizes images to 256x256 pixels and handles SAR grayscale format.
- Normalizes pixel values and applies speckle noise reduction (SAR-specific).
- Data augmentation: horizontal/vertical flips, rotations, brightness, and contrast adjustment. **(Extra)**
- Visualizes dataset samples and computes statistics (mean, std deviation) before/after augmentation.
- Splits dataset into training, validation, and test sets.
- Saves processed datasets as `.npy` files for use in later milestones.

### Milestone 2: Model Development, Training, & Evaluation
- Loads preprocessed datasets saved from milestone 1.
- Builds and trains a Convolutional Neural Network (CNN) for binary classification (oil spill vs no spill).
- CNN architecture includes dropout layers for better generalization. **(Extra)**
- Uses Adam optimizer with a tunable learning rate.
- Incorporates early stopping and model checkpointing to minimize overfitting and retain best weights. **(Extra)**
- Visualizes training/validation accuracy and loss curves.
- Prints classification report and confusion matrix for comprehensive evaluation. **(Extra)**

---

## How To Run

### Prerequisites
- Python 3.x
- Required libraries:
    ```
    pip install numpy matplotlib opencv-python scikit-learn tensorflow
    ```

### Step 1: Run Milestone 1
1. Clone this repository.
2. In Google Colab or locally, run:
    ```
    python Milestone_1.py
    ```
   This will mount Google Drive, preprocess images, perform augmentation, and save train/val/test sets (`.npy` files).

### Step 2: Run Milestone 2
1. Ensure processed datasets from milestone 1 (`train_images.npy`, etc.) are present in the working directory.
2. Run:
    ```
    python Milestone_2.py
    ```
   This will train and evaluate the CNN model.

---

## Notes
- Features marked **(Extra)** are included to improve efficiency and robustness far beyond basic requirements.
- Refer to the detailed comments in each code file for further explanations.

---

## Author

- Name: Jeevietha
- Email: jeevietha11@gmail.com
- GitHub: https://github.com/jeev170

---

Feel free to contribute or open Issues for feedback and suggestions!
