import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


# Load trained model
model = tf.keras.models.load_model(
    "models/pneumonia_model.keras"
)

print("Model Loaded Successfully!")


# X-ray image path
img_path = "dataset/chest_xray/test/PNEUMONIA/person99_bacteria_474.jpeg"
# Preprocess image
img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0


# Prediction
prediction = model.predict(img_array)


score = prediction[0][0]


# Result
if score >= 0.5:
    print("\nPrediction: PNEUMONIA")
    print(f"Confidence: {score*100:.2f}%")

else:
    print("\nPrediction: NORMAL")
    print(f"Confidence: {(1-score)*100:.2f}%")