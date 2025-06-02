import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# === Avaliação final ===
loss, acc = model.evaluate(val_data)
print(f"Acurácia em validação: {acc:.2f}")

# === Gráficos de desempenho ===
plt.figure(figsize=(12, 5))

# Acurácia por época
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

# Perda por época
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda (Loss) por época')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# === Matriz de Confusão ===

# Rótulos verdadeiros e previstos
y_true = val_data.classes
y_pred_probs = model.predict(val_data)
y_pred = np.argmax(y_pred_probs, axis=1)

# Geração e exibição da matriz
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_usadas)

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Matriz de Confusão - Validação")
plt.grid(False)
plt.show()
