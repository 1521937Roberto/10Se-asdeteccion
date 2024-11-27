import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp

# Cargar el modelo entrenado
model = load_model("modelo_lenguaje_senas_entrenado.h5")

# Definir tamaño de imagen de entrada (debe coincidir con el tamaño de las imágenes utilizadas para entrenar)
IMG_SIZE = 128

# Inicializar Mediapipe Hands para la detección de la mano
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Función para preprocesar la imagen (igual que el preprocesamiento durante el entrenamiento)
def preprocesar_imagen(frame):
    # Convertir a RGB (Mediapipe espera imágenes en este formato)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen para obtener los puntos clave de la mano
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Solo usamos el primer conjunto de puntos (en caso de haber varias manos)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extraer las coordenadas de los puntos de la mano (normalizadas)
        puntos = []
        for lm in hand_landmarks.landmark:
            puntos.append([lm.x, lm.y, lm.z])  # x, y, z de cada punto clave
        
        # Crear una imagen en blanco de tamaño 128x128 para dibujar los puntos
        imagen = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Dibujar los puntos clave sobre la imagen
        for i, lm in enumerate(puntos):
            x = int(lm[0] * IMG_SIZE)  # Normalizar las coordenadas
            y = int(lm[1] * IMG_SIZE)
            cv2.circle(imagen, (x, y), 3, (255, 255, 255), -1)  # Dibujar el punto en blanco
        
        # Redimensionar la imagen a 128x128 y normalizarla (como durante el entrenamiento)
        imagen = cv2.resize(imagen, (IMG_SIZE, IMG_SIZE))
        imagen = imagen.astype('float32') / 255.0
        imagen = np.expand_dims(imagen, axis=0)  # Agregar dimensión para el batch
        
        return imagen
    return None

# Función para predecir el gesto
def predecir_gesto(imagen):
    # Hacer la predicción
    pred = model.predict(imagen)
    pred_index = np.argmax(pred)  # Obtener el índice de la clase con mayor probabilidad
    
    # Obtener el nombre del gesto a partir del índice (según el orden de las clases del entrenamiento)
    gestos = list(model.class_indices.keys())  # Obtener el orden de las clases del modelo
    return gestos[pred_index]

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Iniciar ciclo de detección
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Voltear la imagen para mostrarla como en un espejo
    frame = cv2.flip(frame, 1)

    # Preprocesar la imagen para la predicción
    imagen = preprocesar_imagen(frame)
    
    if imagen is not None:
        # Predecir el gesto usando el modelo entrenado
        gesto_predicho = predecir_gesto(imagen)

        # Mostrar el gesto predicho en la imagen
        cv2.putText(frame, f"Gesto: {gesto_predicho}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar la imagen en tiempo real
    cv2.imshow("Detección de Gestos", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
