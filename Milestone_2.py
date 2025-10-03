import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# -------- EXTRA FEATURE: Real-time data augmentation ---------
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Dice loss
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def IoU(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# --------- EXTRA FEATURE: U-Net with BatchNormalization and Dropout ---------
def unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1) # EXTRA FEATURE
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2) # EXTRA FEATURE
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3) # EXTRA FEATURE
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.3)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = BatchNormalization()(c5) # EXTRA FEATURE
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6) # EXTRA FEATURE
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7) # EXTRA FEATURE
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def plot_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    if 'dice_coefficient' in history.history:
        plt.plot(history.history['dice_coefficient'], label='Train Dice')
        plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
        plt.legend()
    plt.title('Dice Coefficient')
    plt.show()

def main():
    # ---- Load data (images and masks should have shape [samples, 256, 256, 1]) ----
    train_imgs = np.load('train_images.npy')
    train_masks = np.load('train_masks.npy')
    val_imgs = np.load('val_images.npy')
    val_masks = np.load('val_masks.npy')
    test_imgs = np.load('test_images.npy')
    test_masks = np.load('test_masks.npy')

    # ---- EXTRA FEATURE: Real-time augmentation with Keras ----
    image_datagen = ImageDataGenerator(rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.07,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    mask_datagen = ImageDataGenerator(rotation_range=15,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      shear_range=0.07,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

    # Fit the generators to images/masks
    seed = 42
    image_datagen.fit(train_imgs, seed=seed)
    mask_datagen.fit(train_masks, seed=seed)

    # Data generator
    def train_generator(batch_size):
        img_gen = image_datagen.flow(train_imgs, batch_size=batch_size, seed=seed)
        mask_gen = mask_datagen.flow(train_masks, batch_size=batch_size, seed=seed)
        while True:
            imgs = img_gen.next()
            masks = mask_gen.next()
            yield imgs, masks

    # ---- Build and compile U-Net ----
    model = unet(input_shape=(256,256,1))
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss=bce_dice_loss,
                  metrics=[dice_coefficient, tf.keras.metrics.BinaryAccuracy()])
    
    model.summary()

    # ---- EXTRA FEATURES: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau ----
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('unet_best_model.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.7, min_lr=1e-6) # EXTRA FEATURE

    batch_size = 16
    steps_per_epoch = len(train_imgs) // batch_size
    validation_steps = len(val_imgs) // batch_size

    history = model.fit(train_generator(batch_size),
                        validation_data=(val_imgs, val_masks),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        epochs=50,
                        callbacks=[early_stop, checkpoint, reduce_lr])

    plot_history(history)

    # ---- Evaluate metrics on test set ----
    preds = model.predict(test_imgs)
    preds_bin = (preds > 0.5).astype(np.uint8)

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(len(test_masks)):
        y_true = test_masks[i].flatten()
        y_pred = preds_bin[i].flatten()
        dice_score = (2 * np.sum(y_true * y_pred) + 1e-6) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)
        iou_score = (np.sum(y_true * y_pred) + 1e-6) / (np.sum(y_true) + np.sum(y_pred) - np.sum(y_true * y_pred) + 1e-6)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        dice_scores.append(dice_score)
        iou_scores.append(iou_score)
        precision_scores.append(precision)
        recall_scores.append(recall)

    print(f"Test Dice Coefficient: {np.mean(dice_scores):.4f}")
    print(f"Test IoU: {np.mean(iou_scores):.4f}")
    print(f"Test Precision: {np.mean(precision_scores):.4f}")
    print(f"Test Recall: {np.mean(recall_scores):.4f}")

    # ---- EXTRA FEATURE: Save visual results ----
    for i in range(3): # Save first 3 test results
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(test_imgs[i].squeeze(), cmap='gray')
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(test_masks[i].squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(preds_bin[i].squeeze(), cmap='gray')
        plt.title('Prediction')
        plt.savefig(f'segmentation_result_{i}.png')
        plt.close()

if __name__ == "__main__":
    main()
