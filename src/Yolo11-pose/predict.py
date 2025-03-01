from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n-pose.pt")  # load an official model
model = YOLO("best_11n.pt")  # load a custom model

# Predict with the model
results = model("dataset/images/val/ytz1-896.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    clf = result.probs 
    result.show()