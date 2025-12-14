from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('500photosv11_200epochs_modified.pt')

# Run inference on the source 
results =model(source=0, show=True, conf=0.3, save=True) #generator of Results
