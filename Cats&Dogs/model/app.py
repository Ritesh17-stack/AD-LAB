import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import joblib
from keras.models import load_model

# Load models
cnn_model = load_model('conv.h5')
kmeans_model = joblib.load('kmeans.joblib')
lr_model = joblib.load('lr_model.joblib')
rf_model = joblib.load('rf_model.joblib')
svm_model = joblib.load('svm_model.joblib')

# Placeholder functions for model predictions
def predict_with_cnn(image_path):
    image = Image.open(image_path).resize((100, 100))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = image_array.reshape(1, 100, 100, 3)
    prediction = cnn_model.predict(image_array)
    return "Cat" if prediction[0][0] > 0.5 else "Dog"

def predict_with_logistic_regression(image_path):
    image = Image.open(image_path).resize((300, 100)).convert('L')  # Resize to 300x100
    image_array = np.array(image).reshape(1, 30000).astype('float32') / 255.0  # Flatten to match 30000 features
    prediction = lr_model.predict(image_array)
    return "Cat" if prediction[0] == 0 else "Dog"

def predict_with_kmeans(image_path):
    image = Image.open(image_path).resize((100, 100)).convert('L')
    image_array = np.array(image).reshape(1, -1).astype('float32') / 255.0
    cluster = kmeans_model.predict(image_array)
    return "Cat" if cluster[0] == 0 else "Dog"

def predict_with_random_forest(image_path):
    image = Image.open(image_path).resize((100, 100)).convert('L')
    image_array = np.array(image).reshape(1, -1).astype('float32') / 255.0
    prediction = rf_model.predict(image_array)
    return "Cat" if prediction[0] == 0 else "Dog"

def predict_with_svm(image_path):
    image = Image.open(image_path).resize((100, 100)).convert('L')
    image_array = np.array(image).reshape(1, -1).astype('float32') / 255.0
    prediction = svm_model.predict(image_array)
    return "Cat" if prediction[0] == 0 else "Dog"

# Mapping model names to prediction functions
model_functions = {
    "cnn": predict_with_cnn,
    "logistic_regression": predict_with_logistic_regression,
    "kmeans": predict_with_kmeans,
    "random_forest": predict_with_random_forest,
    "svm": predict_with_svm
}

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get selected model
        selected_model = request.form.get('model')
        if selected_model not in model_functions:
            return "Invalid model selected"

        # Predict using the selected model
        prediction_function = model_functions[selected_model]
        prediction = prediction_function(filepath)

        return render_template('result.html', prediction=prediction, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
