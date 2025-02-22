import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
IMAGE_FOLDERS = [
    os.path.join("D:\ML model\HAM10000_images_part_1"), 
    os.path.join( "D:\ML model\HAM10000_images_part_2"), 
    os.path.join( "D:\ML model\HAM10000_segmentations_lesion_tschandl"),
]
CSV_FILE = os.path.join("D:\ML model\HAM10000_metadata")
# 2️⃣ Load Labels
df = pd.read_csv(CSV_FILE)
df = df[["image_id", "dx"]]
benign_classes = ["bkl", "df", "nv", "vasc"]
df["label"] = df["dx"].apply(lambda x: 0 if x in benign_classes else 1)
image_size = (128, 128)  # Resize images
X, y = [], []
for folder in IMAGE_FOLDERS:
    print(f"📂 Processing folder {folder}...")
    if not os.path.exists(folder):  
        print(f"⚠️ Warning: Folder {folder} not found. Skipping...")
        continue  # Skip missing folders

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_id = os.path.splitext(filename)[0]  # Remove extension
            if img_id in df["image_id"].values:  # Match CSV
                try:
                    label = df[df["image_id"] == img_id]["label"].values[0]
                    img = load_img(img_path, target_size=image_size)
                    img_array = img_to_array(img) / 255.0  # Normalize
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")

X = np.array(X)
y = np.array(y)
print(f"✅ Total images loaded: {len(X)}")
print(f"✅ Total labels loaded: {len(y)}")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5️⃣ Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)
# 4️⃣ Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 6️⃣ Load Pre-Trained MobileNetV2 Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze pre-trained layers

# 7️⃣ Add Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    validation_data=(X_val, y_val), 
                    epochs=30)


# 9️⃣ Save Model
model.save(os.path.join(BASE_DIR, "skin_cancer_mobilenetv2.h5"))

# 🔟 Plot Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
