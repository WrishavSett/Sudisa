import cv2
import ultralytics
from ultralytics import YOLO

model = YOLO("G:/My Drive/SUDISA/dev/runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(input("Enter video path: "))
out = "output.mp4"

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(out, fourcc, fps, (frame_width, frame_height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)
    print(results)
    break

    # # Annotate frame
    # annotated_frame = results.render()[0]

    # # Display annotated frame
    # cv2.imshow('YOLO Inference', annotated_frame)
    # out.write(annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()