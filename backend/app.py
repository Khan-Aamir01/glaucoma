from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import random
from flask_cors import CORS  
from pipeline import run_glaucoma_inference

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Run actual AI inference 👇
        result = run_glaucoma_inference(filepath)
        print(result)
        return jsonify({
            "filename": filename,
            "decision": result["decision"],
            "confidence": result["confidence"],
            "risk_score": result["risk_score"],
            "explanation": result["explanation"],
            "model_outputs": {
                "resnet_prob": result["model_outputs"]["resnet_prob"],
                "yolo_quality": result["model_outputs"]["yolo_quality"],
                "vessel_density": result["model_outputs"]["vessel_density"],
            },
            "message": "Glaucoma detection complete ✅"
        })

    except Exception as e:
        print("Error during inference:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
