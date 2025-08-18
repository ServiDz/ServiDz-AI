import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Settings
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# 2. Prepare Data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. Simple Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# 4. Train
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# 5. Save
model.save('servidz_model.h5')
print(f"Model saved! Found {train_generator.num_classes} service categories.")