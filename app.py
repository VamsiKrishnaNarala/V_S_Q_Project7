from flask import Flask, render_template, request, redirect, url_for
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import os

app = Flask(__name__)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        # Get the uploaded file and question
        image_file = request.files.get("image")
        question = request.form.get("question")

        if image_file and question:
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Process the image and question
            raw_image = Image.open(image_path).convert("RGB")
            inputs = processor(raw_image, question, return_tensors="pt")
            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)

            return render_template("index.html", answer=answer, image_url=image_path, question=question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
