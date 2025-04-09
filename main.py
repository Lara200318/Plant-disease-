  !pip install -q tensorflow gradio

# Step 2: Import libraries
import tensorflow as tf
import gradio as gr
import numpy as np
import zipfile
import os
import shutil
from PIL import Image
from google.colab import files
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Step 3: Upload ZIP
uploaded = files.upload()
zip_file = list(uploaded.keys())[0]

# Step 4: Extract ZIP
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("dataset")

# Step 5: Remove non-image files (like .DS_Store etc.)
import imghdr
for root, dirs, files in os.walk("dataset"):
    for fname in files:
        fpath = os.path.join(root, fname)
        if imghdr.what(fpath) is None:
            print("❌ Removing non-image file:", fpath)
            os.remove(fpath)

# Step 6: Prepare dataset
img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 7: Build model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 8: Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Step 9: Save model
model.save("plant_disease_model.h5")

# Step 10: Reload model & predict
model = tf.keras.models.load_model("plant_disease_model.h5")
class_names = list(train_generator.class_indices.keys())

def predict(img):
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)[0]
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

# Step 11: Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🌿 Plant Disease Detector",
    description="Upload a leaf image to detect the disease."
)

interface.launch(share=True)
