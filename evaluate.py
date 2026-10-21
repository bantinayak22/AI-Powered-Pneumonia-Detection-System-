import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. Load trained model
# ==========================================

model = tf.keras.models.load_model(
    "models/pneumonia_model.keras"
)


# ==========================================
# 2. Test dataset
# ==========================================

test_dir = "dataset/chest_xray/test"


test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


# ==========================================
# 3. Model prediction
# ==========================================

predictions = model.predict(
    test_generator,
    verbose=1
)


# Convert probabilities to classes
predicted_classes = (
    predictions.ravel() >= 0.5
).astype(int)


# Actual classes
actual_classes = test_generator.classes


# ==========================================
# 4. Accuracy
# ==========================================

accuracy = accuracy_score(
    actual_classes,
    predicted_classes
)


print("\n================================")
print("      MODEL EVALUATION RESULTS")
print("================================")


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 5. Classification Report
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=[
            "NORMAL",
            "PNEUMONIA"
        ]
    )
)


# ==========================================
# 6. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    actual_classes,
    predicted_classes
)


print("\nConfusion Matrix:\n")

print(cm)


# ==========================================
# 7. Save Confusion Matrix Image
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "NORMAL",
        "PNEUMONIA"
    ]
)


disp.plot(
    values_format="d"
)


plt.title(
    "Pneumonia Detection - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    "confusion_matrix.png",
    dpi=300
)


plt.show()


print(
    "\nConfusion matrix saved as: "
    "confusion_matrix.png"
)