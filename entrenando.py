from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('500photos.pt')

# Run inference on the source 
results =model(source=0, show=True, conf=0.3, save=True) #generator of Results