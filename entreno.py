import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. Definir parámetros de entrada
IMG_SIZE = 128  # Tamaño de las imágenes de entrada
BATCH_SIZE = 32
EPOCHS = 20  # Puedes ajustar esto según sea necesario

# Definir la carpeta base donde están las imágenes (debe contener subcarpetas con nombres de gestos)
dataset_dir = "dataset_abecedario"

# 2. Preprocesamiento de imágenes utilizando ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizar las imágenes a un rango [0, 1]
    validation_split=0.2  # Usar 20% para validación
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),  # Redimensionar todas las imágenes
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # Usamos 'categorical' porque tenemos varias clases
    subset="training",  # Para el conjunto de entrenamiento
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),  # Redimensionar todas las imágenes
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # Usamos 'categorical' porque tenemos varias clases
    subset="validation",  # Para el conjunto de validación
)

# 3. Crear el modelo CNN
model = Sequential()

# Capa convolucional
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))

# Capa convolucional adicional
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Aplanar la salida de las capas convolucionales
model.add(Flatten())

# Capa densa (fully connected)
model.add(Dense(128, activation='relu'))

# Capa de Dropout para evitar sobreajuste
model.add(Dropout(0.5))

# Capa de salida con softmax para clasificación multiclase
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

# Compilar el modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Ver resumen del modelo
model.summary()

# 4. Entrenamiento del modelo

# Usar EarlyStopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=2,
    callbacks=[early_stopping]
)

# 5. Guardar el modelo entrenado
model.save("modelo_lenguaje_senas_entrenado.h5")
print("Modelo guardado como 'modelo_lenguaje_senas_entrenado.h5'")

# 6. Evaluar el modelo (opcional)
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validación - Pérdida: {val_loss}, Precisión: {val_acc}")
