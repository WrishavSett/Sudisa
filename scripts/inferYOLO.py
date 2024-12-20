import glob
from ultralytics import YOLO
# img_path = input("Enter image path: ")
model = YOLO('D:\SUDISA\SudisaItem12K.pt')
print(model.names)

# results = model.predict(source=img_path, conf=0.4)
#
# for result in results:
#   boxes = result.boxes
#   masks = result.masks
#   keypoints = result.keypoints
#   probs = result.probs
#   obb = result.obb
#   result.show()