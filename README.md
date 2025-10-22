# 🌊 Oil Spill Segmentation with U-Net ⚓

A deep learning pipeline for oil spill detection and semantic segmentation on water bodies using a **U-Net** architecture. This project encompasses model training, hyperparameter tuning, evaluation, visual reporting, and an interactive **Streamlit** web application for demonstration and real-world file testing.

---

## 🎯 Project Overview

This repository implements **U-Net-based semantic segmentation** for identifying and segmenting oil spills from satellite or aerial imagery.

### Features:
- **Model Training & Hyperparameter Tuning** using KerasTuner (Hyperband algorithm).
- **Automated Evaluation:** Generation of prediction overlays, calculation of segmentation metrics (**Dice Coefficient, IoU, Pixel Accuracy**), and confusion matrix visualization.
- **Interactive Streamlit Web Demo** for easy visual inspection, downloading of results, and direct comparison of predictions.

---

## 📂 Project Structure

The key files and directories are structured as follows:

.
├── README.md
├── evaluate_and_visualize.py         # Model evaluation & visualization script
├── hyperband_tuner.py                # Hyperparameter tuning (KerasTuner)
├── model_define_train.py             # U-Net model definition and training logic
├── setup_and_load.py                 # Data preparation/util scripts
├── streamlit_app_launcher.py         # Streamlit app for interactive demo
├── training_history_loss.png         # Training/validation loss curve
├── unet_architecture.json            # Saved U-Net model architecture
├── unet_best_weights.weights.h5      # Best trained model weights
└── unet_history.json                 # Training history and metrics

---

## ⚙️ Requirements

### Software
- **Python** `v3.8` or higher
- **TensorFlow** `v2.0` or higher

### Dependencies
Install all necessary packages using `pip`:

pip install tensorflow keras keras-tuner scikit-learn matplotlib seaborn numpy imageio pillow streamlit

---

## 📊 Dataset Preparation

Organize your oil spill segmentation dataset into the following structure:

oil_spill_dataset/
├── train/
│   ├── images/ : input images
│   └── masks/  : binary segmentation masks

The `setup_and_load.py` script handles data loading and preprocessing from this structure.

---

## 🚀 Training & Tuning

### Model Training
To define, train, and save the U-Net model:

python model_define_train.py

### Hyperparameter Tuning
To run the Hyperband tuner using KerasTuner to find the optimal hyperparameters:

python hyperband_tuner.py

---

## 🔬 Evaluation & Visualization

Generate predictions, visualization overlays, segmentation metrics, and a confusion matrix by running:

python evaluate_and_visualize.py

Results and visualizations will be saved to your specified output directory (e.g., `oil_spill_outputs/`).

---

## 💻 Streamlit Web App

Launch the interactive demonstration web application for easy visual testing and result inspection:

streamlit run streamlit_app_launcher.py

### App Features:
- Upload or select images directly from the dataset.
- Visualize the **Input Image**, **Predicted Mask**, and the **Overlay** side-by-side (all resized to 256×256).
- View segmentation metrics in real-time.
- Download the generated prediction results.
- See the confusion matrix rendered for the specific image.
- User-friendly interface with sidebar alerts and progress indicators.

---

## 📦 Files Description

| File | Description |
| :--- | :--- |
| `model_define_train.py` | Core script for defining, training, and saving the U-Net model. |
| `hyperband_tuner.py` | Executes the hyperparameter search using KerasTuner. |
| `evaluate_and_visualize.py` | Generates prediction visualizations, overlays, metrics, and confusion matrix. |
| `setup_and_load.py` | Utility functions for data setup, loading, and preprocessing. |
| `streamlit_app_launcher.py` | Main script to launch the interactive Streamlit web application. |
| `unet_best_weights.weights.h5` | The best performing model weights saved after training. |
| `unet_architecture.json` | JSON file containing the structure of the U-Net model. |
| `training_history_loss.png` | Plot of training and validation loss over epochs. |
| `unet_history.json` | JSON file containing the training history and metrics. |

---

## 🖼️ Results Samples

Outputs and results are found in **`oil_spill_outputs/`** or **`streamlit_user_outputs/`**, including overlays, predicted masks, and confusion matrices.
