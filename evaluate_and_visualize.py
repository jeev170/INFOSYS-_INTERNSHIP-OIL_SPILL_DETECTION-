import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import imageio
import json
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import model_from_json


# Custom metrics with registration to enable loading
@register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


DRIVE_BASE = r"C:\Users\jeevi\Downloads\OIL"
OUTPUT_DIR = os.path.join(DRIVE_BASE, "oil_spill_outputs")

# Load data
X_val = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(OUTPUT_DIR, "Y_val.npy"))

# Load model architecture and weights
with open(os.path.join(OUTPUT_DIR, "unet_architecture.json"), "r") as f:
    model_json = f.read()
unet = model_from_json(model_json)

weights_path = os.path.join(OUTPUT_DIR, "unet_best_weights.h5")
if os.path.exists(weights_path):
    unet.load_weights(weights_path)
    print(" Loaded trained weights.")
else:
    print(" Weights file not found.")

# Compile with custom metrics for evaluation
unet.compile(optimizer="adam",
             loss="binary_crossentropy",
             metrics=["accuracy", iou_metric, dice_coefficient])

# Predict and threshold
preds = unet.predict(X_val, batch_size=8)
preds_binary = (preds > 0.5).astype(np.uint8)

def overlay_mask_on_image(image, mask, color=(1, 0, 0), alpha=0.5):
    img = np.clip(image, 0, 1)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    overlay = np.zeros_like(img)
    overlay[mask.astype(bool)] = color
    return img * (1 - alpha) + overlay * alpha

n_show = min(5, len(X_val))
idxs = np.random.choice(len(X_val), n_show, replace=False)

# Create output dirs
overlay_dir = os.path.join(OUTPUT_DIR, "example_overlays")
os.makedirs(overlay_dir, exist_ok=True)

# Visualize and save overlays
for i, idx in enumerate(idxs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(X_val[idx])
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(Y_val[idx].squeeze(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_mask_on_image(X_val[idx], preds_binary[idx].squeeze(),
                                    color=(1, 0, 0), alpha=0.4))
    plt.title("Predicted Overlay")
    plt.axis("off")

    save_path = os.path.join(overlay_dir, f"validation_overlay_{i}.png")
    plt.savefig(save_path)
    plt.close()
    print(f" Saved overlay image to {save_path}")

# Confusion matrix plotting and saving
cm = confusion_matrix(Y_val.flatten(), preds_binary.flatten())
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix (Pixel-wise)")
plt.xlabel("Predicted")
plt.ylabel("True")

conf_matrix_path = os.path.join(overlay_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()
print(f" Confusion matrix image saved to {conf_matrix_path}")

# Save overlays as separate PNG masks
for i, idx in enumerate(idxs[:3]):
    overlay_img = (overlay_mask_on_image(X_val[idx], preds_binary[idx].squeeze(),
                                        color=(1, 0, 0), alpha=0.4) * 255).astype(np.uint8)
    path = os.path.join(overlay_dir, f"overlay_mask_{i}.png")
    imageio.imwrite(path, overlay_img)
    print(f" Saved overlay mask image to {path}")

# Save training loss plot if history file exists
history_path = os.path.join(OUTPUT_DIR, "unet_history.json")
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    plt.figure()
    plt.plot(history.get('loss', []), label='Training Loss')
    plt.plot(history.get('val_loss', []), label='Validation Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    history_plot_path = os.path.join(OUTPUT_DIR, "training_history_loss.png")
    plt.savefig(history_plot_path)
    plt.close()
    print(f" Training history plot saved to {history_plot_path}")
else:
    print("Training history JSON not found; skipping loss plot saving.")
