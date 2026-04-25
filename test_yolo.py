from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pre-trained model

results = model("https://ultralytics.com/images/bus.jpg", save=True)

print("Detection done!")