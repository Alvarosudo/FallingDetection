from ultralytics import YOLO
import time
import winsound
import threading
import cv2

# -----------------------------
# Configuración de alerta
# -----------------------------
ALERT_SECONDS = 2
fall_start_time = None
alert_triggered = False

ALERT_SOUND = "alert.wav"  

def play_alert():
    # Beep de 1000Hz durante 700ms
    winsound.Beep(1000, 700)



# -----------------------------
# Cargar modelo YOLOv8
# -----------------------------
model = YOLO('500photos.pt')

# Abrir webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: no se pudo abrir la webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar detección en el frame actual
    results = model.predict(frame, conf=0.3, verbose=False)

    fall_detected = False
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0]) # Obtener clase detectada (caida 0 no caída 1)
            if cls == 0:  # 
                fall_detected = True # Marcar que se detectó una caída
                break

    # Control de tiempo para la alerta
    if fall_detected:
        if fall_start_time is None:
            fall_start_time = time.time() # Iniciar temporizador
        elapsed = time.time() - fall_start_time # Tiempo transcurrido desde la detección de caída
        # Mostrar temporizador en la pantalla
        cv2.putText(frame, f"Tiempo caida: {elapsed:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if elapsed >= ALERT_SECONDS and not alert_triggered:  # Si supera el umbral y no se ha activado la alerta
            print("¡ALERTA! Persona caída detectada por más de 2 segundos") 
            play_alert()  # Llamada directa, no hace falta threading
            alert_triggered = True # Marcar que la alerta ya fue activada
    else:
        fall_start_time = None
        alert_triggered = False
                # Mostrar "Sin caída" en la pantalla
        cv2.putText(frame, "Sin caida", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen con temporizador
    cv2.imshow("Detección de caídas", frame)

    # Salir con tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
