import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Suppress warnings for a clean output
import warnings
warnings.filterwarnings('ignore')

# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available: ", gpus)
else:
    print("No GPU found. Training on CPU.")

# Load the dataset and create a DataFrame with image paths and labels
data_dir = 'C:/Users/sinha/Downloads/data/test'

image_paths = []
labels = []

for label in ['benign', 'malignant']:
    label_dir = os.path.join(data_dir, label)
    for image_file in os.listdir(label_dir):
        image_paths.append(os.path.join(label_dir, image_file))
        labels.append(label)

# Create a DataFrame
df = pd.DataFrame({
    'filepath': image_paths,
    'label_bin': labels
})

# Print the first few file paths and labels
print(df['filepath'].head())

# Check for class distribution
class_counts = df['label_bin'].value_counts()
print(f"Class distribution:\n{class_counts}")

# Plot distribution of classes
x = df['label_bin'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.show()

# Display sample images from each category (benign and malignant)
for label_bin in df['label_bin'].unique():
    temp = df[df['label_bin'] == label_bin]
    index_list = temp.index
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(f'Images for {label_bin} category', fontsize=20)
    for i in range(4):
        index = np.random.randint(0, len(index_list))
        index = index_list[index]
        data = df.iloc[index]
        image_path = data['filepath']
        img = np.array(Image.open(image_path))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_bin'], random_state=42)

# Data Augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,  # Reduced rotation range
    width_shift_range=0.05,  # Reduced shift range
    height_shift_range=0.05,
    shear_range=0.05,  # Reduced shear range
    zoom_range=0.05,  # Reduced zoom range
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Train generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label_bin',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42,
    workers=4,
    max_queue_size=10
)

# Validation generator
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='label_bin',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False,
    seed=42,
    workers=4,
    max_queue_size=10
)

# Define the model (Using EfficientNetB0)
pre_trained_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Unfreeze all layers
for layer in pre_trained_model.layers:
    layer.trainable = True

# Build the model
inputs = layers.Input(shape=(224, 224, 3))
x = pre_trained_model(inputs, training=True)  # Set training=True to enable fine-tuning
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# Compile the model with an optimized learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Adjusted learning rate

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Callbacks for early stopping, saving the best model, and reducing learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Evaluate the model
val_loss, val_accuracy, val_auc = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation AUC: {val_auc}')

# Plot loss and validation loss
hist_df = pd.DataFrame(history.history)
plt.plot(hist_df['accuracy'], label='Training Accuracy')
plt.plot(hist_df['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(hist_df['loss'], label='Training Loss')
plt.plot(hist_df['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(hist_df['AUC'], label='Training AUC')
plt.plot(hist_df['val_AUC'], label='Validation AUC')
plt.title('AUC vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()
