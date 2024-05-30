# Import necessary libraries
from flask import Flask, request, render_template, send_file
from flask_ngrok import run_with_ngrok
import joblib
import os
import pandas as pd
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)

# Directory where models are saved
model_dir = 'saved_models'
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)

# Function to load a model pipeline
def load_model(model_name):
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise ValueError(f"Model {model_name} is not recognized or does not exist.")

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle file upload and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        data = pd.read_csv(file_path)
        
        # Drop columns named 'id' or 'class' if they exist
        if 'id' in data.columns:
            data = data.drop(columns=['id'])
        if 'class' in data.columns:
            data = data.drop(columns=['class'])
        
        model_name = request.form['model']
        model = load_model(model_name)
        
        predictions = model.predict(data)
        data['Predictions'] = predictions
        
        output_path = os.path.join(upload_folder, 'predictions.csv')
        data.to_csv(output_path, index=False)
        
        return send_file(output_path, as_attachment=True)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)



