import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 📌 **Initialize Flask App**
app = Flask(__name__)

# 📂 **Load Trained Model**
MODEL_PATH = r"D:/ML model/skin_cancer_mobilenetv2_optimized.h5"
model = load_model(MODEL_PATH)

# 📌 **Define Image Preprocessing Function**
def preprocess_image(image_path, target_size=(128, 128)):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# 📌 **Define API Route for Prediction**
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 **Check if image file is provided**
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)  # Save image temporarily

        # 🔹 **Preprocess Image**
        image = preprocess_image(image_path)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # 🔹 **Extract Additional Metadata**
        age = float(request.form.get("age", 50))  # Default 50 if not provided
        sex = float(request.form.get("sex", 0))  # 0: Male, 1: Female
        loc = float(request.form.get("loc", 0))  # Default 0 if not provided

        # 📌 **Prepare Inputs**
        age = np.array([[age]])
        sex = np.array([[sex]])
        loc = np.array([[loc]])

        # 🔥 **Make Prediction**
        prediction = model.predict([image, age, sex, loc])
        probability = float(prediction[0][0])  # Convert to Python float
        confidence_score = round(probability * 100, 2)  # Convert to percentage

        # ✅ **Return Result**
        return jsonify({
            "prediction": probability,
            "confidence_score": f"{confidence_score}%",
            "diagnosis": "Malignant" if probability > 0.5 else "Benign"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 📌 **Run Flask Server**
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
