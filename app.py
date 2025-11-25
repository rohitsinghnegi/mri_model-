import os
import numpy as np
import cv2
import tensorflow as tf
import warnings
from flask import Flask, request, render_template, redirect, jsonify
import gc

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Configuration ---
MODEL_PATHS = {
    'mri': "best_mri_classifier.h5",
    'skin': "best_skin_classifier.h5",
    'xray': "best_xray_classifier.h5"
}

CLASS_NAMES = {
    'mri': ['glioma', 'meningioma', 'notumor', 'pituitary'],
    'skin': ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'],
    'xray': ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
}

MODEL_NAMES = {
    'mri': "Brain Tumor MRI",
    'skin': "Skin Disease",
    'xray': "Lung Disease X-Ray"
}

# Global variable to hold the currently loaded model
current_model = None
current_model_type = None

def get_model(model_type):
    global current_model, current_model_type
    
    if current_model_type == model_type and current_model is not None:
        return current_model
    
    # Clear memory before loading new model
    if current_model is not None:
        del current_model
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"🗑️ Cleared memory for previous model.")

    print(f"⏳ Loading {model_type} model...")
    path = MODEL_PATHS.get(model_type)
    if not path:
        raise ValueError("Invalid model type")
        
    current_model = tf.keras.models.load_model(path)
    current_model_type = model_type
    print(f"✅ {MODEL_NAMES[model_type]} loaded.")
    return current_model

# --- Model-Specific Preprocessing Functions ---
def preprocess_mri(image_stream):
    """
    Preprocessing for MRI model - uses EfficientNetV2 preprocessing.
    This model works correctly with EfficientNetV2 normalization.
    """
    filestr = image_stream.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be read.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)
    print("✓ Using EfficientNetV2 preprocessing for MRI")
    return img_array

def preprocess_skin(image_stream):
    """
    Preprocessing for Skin model - uses standard /255 normalization.
    Medical imaging models are typically trained with [0, 1] range.
    """
    filestr = image_stream.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be read.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img.astype(np.float32) / 255.0  # Standard normalization [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    print("✓ Using standard /255 normalization for Skin")
    return img_array

def preprocess_xray(image_stream):
    """
    Preprocessing for X-Ray model - uses standard /255 normalization.
    Medical imaging models are typically trained with [0, 1] range.
    """
    filestr = image_stream.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be read.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img.astype(np.float32) / 255.0  # Standard normalization [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    print("✓ Using standard /255 normalization for X-Ray")
    return img_array

def get_preprocessing_function(model_type):
    """
    Returns the appropriate preprocessing function for the given model type.
    """
    preprocessing_map = {
        'mri': preprocess_mri,
        'skin': preprocess_skin,
        'xray': preprocess_xray
    }
    return preprocessing_map.get(model_type)


# --- Main Web Page Route ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        model_type = request.form.get('model_type')

        if file and model_type:
            try:
                # Get the appropriate preprocessing function for this model
                preprocess_func = get_preprocessing_function(model_type)
                if preprocess_func is None:
                    raise ValueError(f"Invalid model type: {model_type}")
                
                img = preprocess_func(file)
                
                # Load model on demand
                model = get_model(model_type)
                
                preds = model.predict(img)
                class_names = CLASS_NAMES[model_type]
                model_name = MODEL_NAMES[model_type]

                class_id = np.argmax(preds[0])
                confidence = preds[0][class_id] * 100
                predicted_class = class_names[class_id]

                # Log all predictions for debugging
                print(f"\n{'='*50}")
                print(f"Model: {model_name}")
                print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
                print(f"All predictions:")
                for i, (cls, prob) in enumerate(zip(class_names, preds[0])):
                    print(f"  {cls}: {prob*100:.2f}%")
                print(f"{'='*50}\n")

                result_str = f"Prediction: {predicted_class}"
                conf_str = f"Confidence: {confidence:.2f}%"
                model_str = f"Model Used: {model_name}"

                return render_template('index.html', result=result_str, confidence=conf_str, model_used=model_str)

            except Exception as e:
                print(f"Error: {e}")
                return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html', result=None, confidence=None, model_used=None, error=None)

# --- API Route for Flutter App ---
@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    model_type = request.form.get('model_type')
    if not model_type or model_type not in MODEL_PATHS:
        return jsonify({"error": "Invalid or missing model_type. Choose 'mri', 'skin', or 'xray'."}), 400

    try:
        # Get the appropriate preprocessing function for this model
        preprocess_func = get_preprocessing_function(model_type)
        if preprocess_func is None:
            return jsonify({"error": f"Invalid model type: {model_type}"}), 400
        
        img = preprocess_func(file)
        
        # Load model on demand
        model = get_model(model_type)
        
        preds = model.predict(img)
        class_names = CLASS_NAMES[model_type]
        model_name = MODEL_NAMES[model_type]

        class_id = np.argmax(preds[0])
        confidence = float(preds[0][class_id] * 100)
        predicted_class = class_names[class_id]

        # Log all predictions for debugging
        print(f"\n{'='*50}")
        print(f"API Request - Model: {model_name}")
        print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
        print(f"All predictions:")
        for i, (cls, prob) in enumerate(zip(class_names, preds[0])):
            print(f"  {cls}: {prob*100:.2f}%")
        print(f"{'='*50}\n")

        return jsonify({
            "model_used": model_name,
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')