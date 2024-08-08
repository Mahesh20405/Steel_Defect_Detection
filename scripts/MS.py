import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('defect_classifier.keras')

# Prediction function
def pred_def(image):
    img = cv2.resize(image, (256, 256))
    img = img.reshape((1, 256, 256, 3)) / 255.0
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return class_idx, prediction

# Implementing camera feed prediction
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    class_idx, confidence = pred_def(frame)
    # Display result
    cv2.putText(frame, f"Class: {class_idx}, Confidence: {confidence[0][class_idx]:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Defect Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
