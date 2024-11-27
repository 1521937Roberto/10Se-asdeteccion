import cv2
import os
import mediapipe as mp

# Configuración inicial
GESTOS = ["A", "1", "2", "3", "4", "5", "OK", "Orejas", "X", "Paz y amor"]  # Selección de letras y gestos
CAPTURAS_POR_GESTO = 200  # Número de fotos por cada gesto
carpeta_base = "dataset_abecedario"  # Carpeta raíz para guardar las imágenes

# Crear carpetas para cada gesto
for gesto in GESTOS:
    ruta_carpeta = os.path.join(carpeta_base, gesto)
    os.makedirs(ruta_carpeta, exist_ok=True)

# Inicializar Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

gesto_actual = 0  # Índice de la letra o gesto actual
capturas_realizadas = 0  # Contador de capturas realizadas para el gesto actual

print(f"Presiona la tecla 'c' para capturar imágenes de la letra o gesto actual ({GESTOS[gesto_actual]}).")
print("Usa las teclas izquierda/derecha para cambiar entre letras/gestos.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Voltear horizontalmente la imagen (modo espejo)
    frame = cv2.flip(frame, 1)

    # Convertir a RGB para Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el marco con Mediapipe Hands
    results = hands.process(frame_rgb)

    # No dibujar puntos clave ni líneas
    # Si quieres hacer algún procesamiento de los puntos, lo puedes hacer, pero no los mostraré.

    # Mostrar el marco actual con instrucciones
    cv2.putText(frame, f"Gesto: {GESTOS[gesto_actual]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Capturas: {capturas_realizadas}/{CAPTURAS_POR_GESTO}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Presiona 'c' para capturar", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Captura de gestos", frame)

    # Capturar teclas presionadas
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Salir con la tecla 'q'
        break
    elif key == ord('c'):  # Capturar imagen con la tecla 'c'
        ruta_guardado = os.path.join(carpeta_base, GESTOS[gesto_actual], f"{GESTOS[gesto_actual]}_{capturas_realizadas}.jpg")
        cv2.imwrite(ruta_guardado, frame)
        print(f"Imagen guardada en: {ruta_guardado}")
        capturas_realizadas += 1
        if capturas_realizadas >= CAPTURAS_POR_GESTO:
            print(f"Capturas completas para el gesto '{GESTOS[gesto_actual]}'. Cambia de gesto para continuar.")
    elif key == ord('d'):  # Cambiar a la siguiente letra/gesto
        gesto_actual = (gesto_actual + 1) % len(GESTOS)
        capturas_realizadas = 0
        print(f"Cambiado al gesto: {GESTOS[gesto_actual]}")
    elif key == ord('a'):  # Cambiar al gesto anterior
        gesto_actual = (gesto_actual - 1) % len(GESTOS)
        capturas_realizadas = 0
        print(f"Cambiado al gesto: {GESTOS[gesto_actual]}")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
