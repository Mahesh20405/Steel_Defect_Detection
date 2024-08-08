from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('E:/PJ/AI VT12/model.keras')

# Function to predict defect class from image
def pred_def(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image as the model was trained with rescaled images

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# Path to test image
test_img_path = 'E:/PJ/AI VT/test_images/0ad9a817d.jpg'  # Replace with the path to the image you want to test

# Predict the defect
predicted_class = pred_def(test_img_path, model)
print(f'The predicted defect class is: {predicted_class}')
