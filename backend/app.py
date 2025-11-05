from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import cv2
from pipeline import run_glaucoma_inference
import segmentation_models_pytorch as smp
import numpy as np

from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  

# ================== CDR MODEL SETUP ==================
CDR_MODEL_PATH = "models/cdr_calculator_best.pth"

# Define same U-Net architecture used in training
cdr_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
    activation=None
)

# Load checkpoint
checkpoint = torch.load(CDR_MODEL_PATH, map_location="cpu",weights_only=False)
cdr_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
cdr_model.eval()

# Preprocessor
class CDRPreprocessor:
    def __init__(self, target_size=256):  # changed to 256
        self.target_size = target_size

    def preprocess(self, image):
        # Resize to 256x256
        img = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        
        # Extract green channel
        green_channel = img[:, :, 1] if len(img.shape) == 3 else img
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        
        # Normalize to 0-1
        normalized = enhanced.astype(np.float32) / 255.0
        
        # Convert to 3 channels
        img_3ch = np.stack([normalized]*3, axis=-1)
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_3ch = (img_3ch - mean) / std

        return img_3ch  # ✅ make sure to return the processed image

preprocessor = CDRPreprocessor(target_size=256)

def calculate_cdr(image_path):
    """Run CDR model and return area-based CDR"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = preprocessor.preprocess(image)
    input_tensor = torch.tensor(input_img).permute(2,0,1).unsqueeze(0).float()  # BCHW

    with torch.no_grad():
        outputs = cdr_model(input_tensor)
        preds = torch.argmax(outputs, dim=1).squeeze(0).numpy()  # Segmentation mask

    # Compute OD and OC areas
    od_area = np.sum(preds == 1)
    oc_area = np.sum(preds == 2)
    cdr_value = (oc_area / od_area)/255 if od_area > 0 else 0.0
    return round(float(cdr_value), 3)

# ================== UPLOAD API ==================
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
        # Run glaucoma model
        result = run_glaucoma_inference(filepath)

        # Run CDR
        cdr_value = calculate_cdr(filepath)
        
        return jsonify({
            "filename": filename,
            "decision": result["decision"],
            "confidence": result["confidence"],
            "risk_score": result["risk_score"],
            "cdr_value": cdr_value,
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
