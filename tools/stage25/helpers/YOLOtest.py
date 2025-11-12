from ultralytics import YOLO
import cv2

# Load the model
model = YOLO(r"C:/Users/dkz/runs/detect/checkbox_optimized/weights/best.pt")

# Run prediction on a test image
results = model.predict(source=r"W:/competition/model/images/test/pltest2_5.jpg", show=True, save=True, conf=0.1)

# Optional: Show results in window
for r in results:
    im_array = r.plot()  # Draw the boxes
    cv2.imshow("Prediction", im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
