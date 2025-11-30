from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder for uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = os.path.join("..", "saved_models", "best_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names from training folders
train_dir = "../dataset/train"
class_names = sorted(os.listdir(train_dir))

def prepare_image(image, target_size=(224,224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    selected_plant = None

    if request.method == "POST":
        file = request.files.get("file")
        selected_plant = request.form.get("plant")
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # URL for HTML
            image_url = f"uploads/{filename}"

            try:
                img = Image.open(save_path)
                img_array = prepare_image(img)
                preds = model.predict(img_array)[0]
                idx = np.argmax(preds)
                prediction = class_names[idx] if idx < len(class_names) else "Unknown"
                confidence = f"{preds[idx]*100:.2f}%"
            except Exception as e:
                prediction = f"Error: {str(e)}"
                confidence = None

    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence, 
                           image_url=image_url,
                           plant_options=sorted(list(set([c.split('___')[0] for c in class_names]))),
                           selected_plant=selected_plant)

if __name__ == "__main__":
    app.run(debug=True)
