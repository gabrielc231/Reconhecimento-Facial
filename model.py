import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Apenas as classes desejadas
classes_usadas = ['angry', 'happy', 'sad', 'surprise']

# Diretório (após separar em pastas por classe)
train_dir = 'data/train'
val_dir = 'data/test'

# Caminho para salvar o modelo
save_path = "models/model6.h5"

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    color_mode='grayscale',
    class_mode='categorical',
    classes=classes_usadas,
    batch_size=32,
    shuffle=True
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(32, 32),
    color_mode='grayscale',
    class_mode='categorical',
    classes=classes_usadas,
    batch_size=32,
    shuffle=False
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.35),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(classes_usadas), activation='softmax')  # n classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stopping]
)


model.save(save_path)
print(f"Modelo salvo com sucesso em: {save_path}")

# Avaliação
loss, acc = model.evaluate(val_data)
print(f"Acurácia em validação: {acc:.2f}")

# Gráfico de acurácia
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.legend()
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Desempenho da CNN')
plt.show()
