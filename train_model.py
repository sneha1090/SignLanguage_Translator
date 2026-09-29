import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------- CONFIG ----------------

DATA_DIR = "data/"
IMG_SIZE = 224
MODEL_PATH = "modelnet_model.h5"
LABELS_PATH = "labels.json"


# ---------------- DATASET ----------------

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# ---------------- LABELS ----------------

labels = list(train_gen.class_indices.keys())

print("\nDetected classes:")
print(labels)

print("\nNumber of classes:", len(labels))

with open(LABELS_PATH, "w") as f:
    json.dump(labels, f, indent=4)

print("Labels saved to:", LABELS_PATH)


# ---------------- MODEL ----------------

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        len(labels),
        activation="softmax"
    )
])


# ---------------- COMPILE ----------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ---------------- TRAIN ----------------

print("\nStarting training...\n")

history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen
)


# ---------------- SAVE ----------------

model.save(MODEL_PATH)

print("\n===================================")
print("Training completed!")
print("Model saved as:", MODEL_PATH)
print("Labels saved as:", LABELS_PATH)
print("===================================")