from flask import Flask, render_template, request, send_from_directory, send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os


app = Flask(__name__)


# ==========================================
# Load trained model
# ==========================================

model = tf.keras.models.load_model(
    "models/pneumonia_model.keras"
)


# ==========================================
# Folders
# ==========================================

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER


# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# ==========================================
# Show uploaded image
# ==========================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ==========================================
# Generate PDF Report
# ==========================================

@app.route("/download_report")
def download_report():

    result = request.args.get("result")
    confidence = request.args.get("confidence")
    image_file = request.args.get("image")

    report_path = os.path.join(
        REPORT_FOLDER,
        "pneumonia_report.pdf"
    )

    pdf = canvas.Canvas(
        report_path,
        pagesize=A4
    )

    width, height = A4


    # ======================================
    # Header
    # ======================================

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawCentredString(
        width / 2,
        height - 60,
        "AI PNEUMONIA DETECTION REPORT"
    )


    pdf.setFont(
        "Helvetica",
        10
    )

    pdf.drawCentredString(
        width / 2,
        height - 80,
        "AI-Powered Chest X-Ray Analysis"
    )


    # ======================================
    # Date and Time
    # ======================================

    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(
        60,
        height - 120,
        f"Report Date & Time: {current_time}"
    )


    # ======================================
    # X-Ray Image
    # ======================================

    if image_file:

        image_path = os.path.join(
            UPLOAD_FOLDER,
            image_file
        )

        if os.path.exists(image_path):

            try:

                img = Image.open(image_path)

                img.thumbnail(
                    (350, 300)
                )

                pdf.drawImage(
                    ImageReader(img),
                    120,
                    height - 470,
                    width=350,
                    height=280,
                    preserveAspectRatio=True,
                    anchor="c"
                )

            except Exception:

                pass


    # ======================================
    # Prediction Section
    # ======================================

    pdf.setFont(
        "Helvetica-Bold",
        15
    )

    pdf.drawString(
        60,
        height - 510,
        "Prediction Result"
    )


    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        60,
        height - 540,
        f"Prediction: {result}"
    )

    pdf.drawString(
        60,
        height - 565,
        f"Confidence: {confidence}%"
    )


    # ======================================
    # AI Findings
    # ======================================

    pdf.setFont(
        "Helvetica-Bold",
        15
    )

    pdf.drawString(
        60,
        height - 610,
        "AI Findings"
    )


    pdf.setFont(
        "Helvetica",
        11
    )


    if result == "PNEUMONIA":

        finding = (
            "The AI model detected patterns in the "
            "uploaded chest X-ray that are associated "
            "with pneumonia."
        )

    else:

        finding = (
            "The AI model did not detect patterns "
            "strongly associated with pneumonia in "
            "the uploaded chest X-ray."
        )


    # Split finding into lines
    words = finding.split()

    line = ""
    y = height - 635

    for word in words:

        test_line = line + " " + word

        if len(test_line) > 85:

            pdf.drawString(
                60,
                y,
                line.strip()
            )

            y -= 18
            line = word

        else:

            line = test_line


    if line:

        pdf.drawString(
            60,
            y,
            line.strip()
        )


    # ======================================
    # Recommendation
    # ======================================

    y -= 45

    pdf.setFont(
        "Helvetica-Bold",
        15
    )

    pdf.drawString(
        60,
        y,
        "Recommendation"
    )


    pdf.setFont(
        "Helvetica",
        11
    )

    y -= 25

    recommendation = (
        "Please consult a qualified medical professional "
        "for proper clinical evaluation. This AI result "
        "should not be used as a standalone diagnosis."
    )


    words = recommendation.split()

    line = ""

    for word in words:

        test_line = line + " " + word

        if len(test_line) > 85:

            pdf.drawString(
                60,
                y,
                line.strip()
            )

            y -= 18
            line = word

        else:

            line = test_line


    if line:

        pdf.drawString(
            60,
            y,
            line.strip()
        )


    # ======================================
    # Disclaimer
    # ======================================

    y -= 50

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        60,
        y,
        "Medical Disclaimer"
    )


    pdf.setFont(
        "Helvetica",
        9
    )

    y -= 20

    disclaimer = (
        "This system is intended for educational and "
        "research purposes only. The AI prediction "
        "does not replace professional medical "
        "diagnosis or medical advice."
    )


    words = disclaimer.split()

    line = ""

    for word in words:

        test_line = line + " " + word

        if len(test_line) > 100:

            pdf.drawString(
                60,
                y,
                line.strip()
            )

            y -= 15
            line = word

        else:

            line = test_line


    if line:

        pdf.drawString(
            60,
            y,
            line.strip()
        )


    # ======================================
    # Footer
    # ======================================

    pdf.setFont(
        "Helvetica",
        8
    )

    pdf.drawCentredString(
        width / 2,
        30,
        "AI Pneumonia Detection System"
    )


    # Save PDF
    pdf.save()


    return send_file(
        report_path,
        as_attachment=True,
        download_name="AI_Pneumonia_Report.pdf"
    )


# ==========================================
# Main Prediction Route
# ==========================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    result = None
    confidence = None
    image_file = None
    report = None


    if request.method == "POST":

        file = request.files.get("image")


        if file and file.filename:

            image_file = file.filename


            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                image_file
            )


            file.save(filepath)


            # ==================================
            # Image Validation
            # ==================================

            try:

                img_check = Image.open(
                    filepath
                )

                img_check.verify()


            except Exception:

                result = "Invalid Image"

                confidence = (
                    "Please upload a valid image"
                )

                return render_template(
                    "index.html",
                    result=result,
                    confidence=confidence,
                    image_file=None,
                    report=None
                )


            # ==================================
            # Image Preprocessing
            # ==================================

            img = image.load_img(
                filepath,
                target_size=(224, 224)
            )


            img_array = image.img_to_array(
                img
            )


            img_array = np.expand_dims(
                img_array,
                axis=0
            )


            img_array = (
                img_array / 255.0
            )


            # ==================================
            # Prediction
            # ==================================

            prediction = model.predict(
                img_array,
                verbose=0
            )


            score = float(
                prediction[0][0]
            )


            # ==================================
            # Classification
            # ==================================

            if score >= 0.5:

                result = "PNEUMONIA"

                confidence = round(
                    score * 100,
                    2
                )


                report = {

                    "title":
                        "AI X-Ray Analysis Report",

                    "finding":
                        "The AI model detected patterns "
                        "associated with pneumonia.",

                    "confidence":
                        f"{confidence}%",

                    "recommendation":
                        "Please consult a qualified "
                        "medical professional for "
                        "proper evaluation."

                }


            else:

                result = "NORMAL"

                confidence = round(
                    (1 - score) * 100,
                    2
                )


                report = {

                    "title":
                        "AI X-Ray Analysis Report",

                    "finding":
                        "The AI model did not detect "
                        "patterns strongly associated "
                        "with pneumonia.",

                    "confidence":
                        f"{confidence}%",

                    "recommendation":
                        "This result is an AI-based "
                        "prediction and should not "
                        "replace professional medical "
                        "evaluation."

                }


    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_file=image_file,
        report=report
    )


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )