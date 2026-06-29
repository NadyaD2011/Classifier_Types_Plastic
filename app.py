import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

CLASS_NAMES = ['PET', 'PE', 'PC', 'PP', 'PS', 'Others']
IMG_SIZE = (224, 224)
MODEL_PATH = 'plastic_type_classifier.keras'

model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Модель загружена")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден в запросе'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    if file:
        try:
            img = Image.open(file.stream).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            predictions = model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            result = {
                'class': CLASS_NAMES[predicted_class_index],
                'confidence': float(confidence)
            }
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Ошибка обработки изображения: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)