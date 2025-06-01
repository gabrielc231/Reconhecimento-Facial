import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

# === GPU Configuration ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === Classes ===
classes_usadas = ['angry', 'happy', 'sad', 'surprise']

# === Diretórios ===
train_dir = 'data/train'
val_dir = 'data/test'

# === Class Weights ===
n_classes = len(classes_usadas)
n_data = [0] * n_classes
for i in range(len(n_data)):
    files = next(os.walk(f"{train_dir}/{classes_usadas[i]}"))[2]
    n_data[i] = len(files)
print("Total de dados por classe:", n_data)

total_data = sum(n_data)
class_weights = {
    i: total_data / (n_data[i] * n_classes) for i in range(n_classes)
}

# === Data Augmentation ===
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    classes=classes_usadas,
    batch_size=64,
    shuffle=True
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    classes=classes_usadas,
    batch_size=64,
    shuffle=False
)

# === Modelo CNN ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.15),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(n_classes, activation='softmax')
])

# === Compilação ===
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# === Callbacks ===
early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# === Treinamento ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stopping, checkpoint],
    class_weight=class_weights
)

# === Salvar Modelo ===
model.save("models/final_model.h5")

# === Avaliação ===
loss, acc = model.evaluate(val_data)
print(f"Acurácia em validação: {acc:.2f}")

# === Gráfico ===
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.legend()
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Desempenho da CNN')
plt.show()
