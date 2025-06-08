from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11s.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/iain/data/yolo_drone_train/data.yaml", epochs=10, batch=32, imgsz=640)

