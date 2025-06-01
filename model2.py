import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Ajustes para usar GPU com limitação de memória
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === Configurações ===
classes_usadas = ['angry', 'happy', 'sad', 'surprise']
train_dir = 'data/train'
val_dir = 'data/test'
save_path = "models/model_mobilenet.h5"
input_size = (96, 96)  # MobileNetV2 requer mínimo 96x96

# === Pré-processamento ===
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=input_size,
    color_mode='rgb',
    class_mode='categorical',
    classes=classes_usadas,
    batch_size=32,
    shuffle=True
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=input_size,
    color_mode='rgb',
    class_mode='categorical',
    classes=classes_usadas,
    batch_size=32,
    shuffle=False
)

# === Base pré-treinada ===
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3)
)
base_model.trainable = False  # "congela" os pesos da base

# === Cabeça personalizada ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(len(classes_usadas), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Compilação ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === Treinamento ===
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stopping]
)

# === Salvando modelo ===
model.save(save_path)
print(f"Modelo salvo com sucesso em: {save_path}")

# === Avaliação e gráfico ===
loss, acc = model.evaluate(val_data)
print(f"Acurácia em validação: {acc:.2f}")

plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.legend()
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Desempenho com Transfer Learning (MobileNetV2)')
plt.show()
