# ============================================================================
# FILE: glaucoma_pipeline.py
# ============================================================================

# ============================================================================
# SECTION 0: IMPORTS
# ============================================================================
import os
import cv2
import numpy as np
import warnings
from pathlib import Path

# ML/Torch Imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================
class Config:
    """Configuration for inference pipeline"""
    
    # --- IMPORTANT ---
    # Update these paths to where your models are stored on the server
    YOLO_MODEL_PATH = 'models/yolov8l_od_oc_best.pt'
    RESNET_MODEL_PATH = 'models/best_resnet_model.pth'
    UNET_VESSEL_MODEL_PATH = 'models/unet_vessel_best.pth'
    
    # Inference parameters
    IMAGE_SIZE = 512
    YOLO_CONF = 0.25
    YOLO_IOU = 0.45
    
    # Decision fusion weights
    WEIGHT_RESNET = 0.60
    WEIGHT_YOLO = 0.20
    WEIGHT_VESSEL = 0.20
    
    # Classification thresholds
    THRESHOLD_NORMAL = 0.35
    THRESHOLD_SUSPICIOUS = 0.65

config = Config()
print("="*60)
print("GLAUCOMA DETECTION PIPELINE - CONFIGURATION")
print("="*60)
print(f"  YOLO model: {config.YOLO_MODEL_PATH}")
print(f"  ResNet model: {config.RESNET_MODEL_PATH}")
print(f"  U-Net Vessel model: {config.UNET_VESSEL_MODEL_PATH}")
print("="*60)

# ============================================================================
# SECTION 2: DEVICE SETUP
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# SECTION 3: PREPROCESSING
# ============================================================================
class FundusPreprocessor:
    """Unified preprocessing pipeline"""
    
    def __init__(self, target_size=512, apply_clahe=True, apply_green_channel=True):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.apply_green_channel = apply_green_channel
    
    def preprocess(self, image):
        """Main preprocessing function"""
        img = cv2.resize(image, (self.target_size, self.target_size), 
                        interpolation=cv2.INTER_AREA)
        
        if self.apply_green_channel and len(img.shape) == 3:
            green_channel = img[:, :, 1]
        else:
            green_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(green_channel)
        else:
            enhanced = green_channel
        
        normalized = enhanced.astype(np.float32) / 255.0
        return np.stack([normalized]*3, axis=-1)

# Initialize preprocessor for U-Net
preprocessor = FundusPreprocessor(target_size=config.IMAGE_SIZE)

# ============================================================================
# SECTION 4: MODEL LOADERS (Global)
# ============================================================================
print("\n" + "="*60)
print("LOADING MODELS... (This happens once at startup)")
print("="*60)

def load_yolo_model(model_path):
    print(f"Loading YOLO from {model_path}...")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    model = YOLO(model_path)
    print("✓ YOLO loaded")
    return model

def load_resnet_model(model_path, num_classes=2):
    print(f"Loading ResNet from {model_path}...")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ResNet model not found: {model_path}")
    
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✓ ResNet loaded")
    return model

def load_unet_vessel_model(model_path):
    print(f"Loading U-Net Vessels from {model_path}...")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"U-Net Vessel model not found: {model_path}")
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ U-Net Vessels loaded")
    return model

# --- Load models into global variables ---
try:
    yolo_model = load_yolo_model(config.YOLO_MODEL_PATH)
    resnet_model = load_resnet_model(config.RESNET_MODEL_PATH)
    unet_vessel_model = load_unet_vessel_model(config.UNET_VESSEL_MODEL_PATH)
    print("\n✅ All models loaded successfully!")
except FileNotFoundError as e:
    print(f"\n❌ FATAL ERROR: {e}")
    print("Pipeline cannot run. Please check model paths in Config class.")
    yolo_model = None
    resnet_model = None
    unet_vessel_model = None
except Exception as e:
    print(f"\n❌ FATAL ERROR during model loading: {e}")
    yolo_model = None
    resnet_model = None
    unet_vessel_model = None

# ============================================================================
# SECTION 5: YOLO INFERENCE
# ============================================================================
def yolo_detect_od_oc(image, model, conf=0.25, iou=0.45):
    results = model(image, conf=conf, iou=iou, verbose=False)
    
    detections = {
        'od_detected': False,
        'oc_detected': False,
        'od_bbox': None,
        'oc_bbox': None,
        'od_conf': 0.0,
        'oc_conf': 0.0,
        'quality_score': 0.0
    }
    
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            if cls == 0:  # Optic Disc
                detections['od_detected'] = True
                detections['od_bbox'] = bbox
                detections['od_conf'] = conf_val
            elif cls == 1:  # Optic Cup
                detections['oc_detected'] = True
                detections['oc_bbox'] = bbox
                detections['oc_conf'] = conf_val
    
    if detections['od_detected'] and detections['oc_detected']:
        detections['quality_score'] = (detections['od_conf'] + detections['oc_conf']) / 2
    elif detections['od_detected']:
        detections['quality_score'] = detections['od_conf'] * 0.5
    else:
        detections['quality_score'] = 0.0
    
    # Return serializable list (or None)
    if detections['od_bbox'] is not None:
        detections['od_bbox'] = detections['od_bbox'].tolist()
    if detections['oc_bbox'] is not None:
        detections['oc_bbox'] = detections['oc_bbox'].tolist()
        
    return detections

# ============================================================================
# SECTION 6: RESNET INFERENCE
# ============================================================================
def resnet_classify(image, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        glaucoma_prob = probs[0, 1].item()
    
    return glaucoma_prob

# ============================================================================
# SECTION 7: U-NET VESSEL SEGMENTATION
# ============================================================================
def unet_segment_vessels(image, model):
    # Use the global preprocessor
    preprocessed = preprocessor.preprocess(image) 
    preprocessed_uint8 = (preprocessed * 255).astype(np.uint8)
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    img_tensor = transform(image=preprocessed_uint8)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred_sigmoid = torch.sigmoid(output).squeeze().cpu().numpy()
        vessel_mask = (pred_sigmoid > 0.5).astype(np.uint8)
    
    vessel_density = np.mean(vessel_mask)
    return vessel_density, vessel_mask

# ============================================================================
# SECTION 8: DECISION FUSION MODULE
# ============================================================================
def fuse_predictions(resnet_prob, yolo_quality, vessel_density):
    vessel_risk = 1.0 - (vessel_density / 0.15)
    vessel_risk = np.clip(vessel_risk, 0.0, 1.0)
    
    final_score = (
        config.WEIGHT_RESNET * resnet_prob +
        config.WEIGHT_YOLO * (1.0 - yolo_quality) +
        config.WEIGHT_VESSEL * vessel_risk
    )
    
    if final_score < config.THRESHOLD_NORMAL:
        decision = "Normal"
        confidence = "High" if final_score < 0.25 else "Moderate"
    elif final_score < config.THRESHOLD_SUSPICIOUS:
        decision = "Suspicious"
        confidence = "Moderate"
    else:
        decision = "Glaucoma"
        confidence = "High" if final_score > 0.75 else "Moderate"
    
    explanation = []
    if resnet_prob > 0.7:
        explanation.append(f"Deep learning classifier indicates HIGH glaucoma risk ({resnet_prob:.2f})")
    elif resnet_prob > 0.5:
        explanation.append(f"Deep learning classifier indicates MODERATE glaucoma risk ({resnet_prob:.2f})")
    else:
        explanation.append(f"Deep learning classifier indicates LOW glaucoma risk ({resnet_prob:.2f})")
    
    if yolo_quality < 0.5:
        explanation.append(f"Image quality is LOW ({yolo_quality:.2f}) - OD/OC detection uncertain")
    
    if vessel_density < 0.08:
        explanation.append(f"Vessel density is LOW ({vessel_density:.3f}) - possible vascular dropout")
    elif vessel_density < 0.10:
        explanation.append(f"Vessel density is BORDERLINE ({vessel_density:.3f})")
    
    return final_score, decision, confidence, explanation


# ============================================================================
# SECTION 9: MAIN INFERENCE FUNCTION (Called by app.py)
# ============================================================================

def run_glaucoma_inference(image_path):
    """
    Runs the complete glaucoma inference pipeline on a single image file.
    
    Args:
        image_path (str): The file path to the uploaded image.
        
    Returns:
        dict: A dictionary containing the full analysis results.
        
    Raises:
        RuntimeError: If models are not loaded.
        ValueError: If the image file cannot be read.
    """
    
    # 1. Check if models are loaded
    if yolo_model is None or resnet_model is None or unet_vessel_model is None:
        print("Error: Models are not loaded. Check pipeline logs.")
        raise RuntimeError("Models are not loaded. Check server logs.")

    try:
        # 2. Read the image from the filepath
        image_bgr = cv2.imread(image_path)
        
        if image_bgr is None:
            raise ValueError(f"Could not read image from path: {image_path}")
        
        # Convert BGR (from OpenCV) to RGB (for models)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 3. Run the full pipeline
        
        # --- Step 1: YOLO Detection ---
        yolo_results = yolo_detect_od_oc(image_rgb, yolo_model, 
                                         conf=config.YOLO_CONF, iou=config.YOLO_IOU)
        
        # --- Step 2: ResNet Classification ---
        preprocessed_img_resnet = preprocessor.preprocess(image_rgb)
        resnet_prob = resnet_classify(preprocessed_img_resnet, resnet_model)
        
        # --- Step 3: U-Net Vessel Segmentation ---
        vessel_density, _ = unet_segment_vessels(image_rgb, unet_vessel_model)
        
        # --- Step 4: Decision Fusion ---
        final_score, decision, confidence, explanation = fuse_predictions(
            resnet_prob, yolo_results['quality_score'], vessel_density
        )

        # 4. Format results dictionary to match app.py
        results = {
            "decision": decision,
            "confidence": confidence,
            "risk_score": float(final_score),
            "explanation": explanation,
            "model_outputs": {
                "resnet_prob": float(resnet_prob),
                "yolo_quality": float(yolo_results['quality_score']),
                "vessel_density": float(vessel_density),
            },
            # You can also include the full yolo_results if needed by the frontend
            "yolo_debug_info": yolo_results 
        }
        
        # 5. Return the Python dictionary
        return results

    except Exception as e:
        print(f"Error during prediction for {image_path}: {e}")
        # Re-raise the exception so app.py's try/except block can catch it
        raise