import os
import re
import io
import base64
import pickle
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open('model/digit_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
        
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    image_bytes = base64.b64decode(image_data)
    
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = ImageOps.invert(img) # Invert to match white on black
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    
    arr = np.array(img, dtype=np.float32) / 16.0
    arr = arr.reshape(1, -1)
    
    probs = model.predict_proba(arr)[0] * 100
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    
    # Format distribution
    dist = [{'digit': i, 'prob': round(float(probs[i]), 1)} for i in range(10)]
    
    return jsonify({
        'prediction': pred,
        'confidence': round(conf, 1),
        'distribution': dist
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
