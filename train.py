from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('V11_bueno.pt')

# Run inference on the source 
# La opción iou cuanto mayor es más bounding boxes genera, en 0.9 se generan muchísimas
# El imgsz cuanto mayor es más bounding boxes obtiene, el óptimo parece ser 640, sino se quita
results =model(source=0, show=True, conf=0.5, iou=0.5, save=True) #generator of Results