from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from flask_cors import CORS
import uuid
import json
from werkzeug.utils import secure_filename
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load clinical model
MODEL_PATH = 'clinical_model.pkl'
SCALER_PATH = 'clinical_scaler.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError('Clinical model or scaler file not found. Please train the clinical model first.')

# Load model data
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
feature_names = model_data['feature_names']
class_names = model_data['class_names']
confidence_thresholds = model_data['confidence_thresholds']
scaler = joblib.load(SCALER_PATH)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_clinical_data(data):
    """Validate clinical data for completeness and ranges"""
    required_features = feature_names
    
    # Check if all required features are present
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return False, f"Missing required features: {missing_features}"
    
    # Validate data types and ranges
    validation_errors = []
    
    # Age validation (should be between 18-100)
    if 'age' in data and (data['age'] < 18 or data['age'] > 100):
        validation_errors.append("Age must be between 18 and 100")
    
    # Binary features validation (family_history, menopausal_status)
    binary_features = ['family_history', 'menopausal_status']
    for feature in binary_features:
        if feature in data and data[feature] not in [0, 1]:
            validation_errors.append(f"{feature} must be 0 or 1")
    
    # Image features validation (should be positive)
    image_features = [f for f in required_features if f not in ['age', 'family_history', 'menopausal_status']]
    for feature in image_features:
        if feature in data and data[feature] <= 0:
            validation_errors.append(f"{feature} must be positive")
    
    if validation_errors:
        return False, f"Validation errors: {'; '.join(validation_errors)}"
    
    return True, "Data validation passed"

def generate_prediction_id():
    """Generate unique prediction ID"""
    return str(uuid.uuid4())

def log_prediction(prediction_id, input_data, result, user_info=None):
    """Log prediction for audit trail"""
    log_entry = {
        'prediction_id': prediction_id,
        'timestamp': datetime.utcnow().isoformat(),
        'input_data_hash': hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest(),
        'result': result,
        'user_info': user_info or 'anonymous'
    }
    
    # In production, this would go to a secure database
    logger.info(f"Prediction logged: {prediction_id}")
    
    # Save to file for demo purposes
    with open('prediction_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': type(model).__name__,
        'feature_count': len(feature_names),
        'features': feature_names,
        'class_names': class_names,
        'confidence_thresholds': confidence_thresholds,
        'training_date': '2024-01-01',  # Would be stored with model
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with clinical validation"""
    prediction_id = generate_prediction_id()
    
    try:
        # Log request
        logger.info(f"Prediction request received: {prediction_id}")
        
        # Get user info from headers (in real app, this would be from authentication)
        user_info = request.headers.get('X-User-ID', 'anonymous')
        
        if request.content_type and request.content_type.startswith('application/json'):
            data = request.get_json()
            
            # Validate single prediction
            if isinstance(data, dict):
                # Validate clinical data
                is_valid, validation_message = validate_clinical_data(data)
                if not is_valid:
                    return jsonify({
                        'error': validation_message,
                        'prediction_id': prediction_id
                    }), 400
                
                # Prepare data for prediction
                X = np.array([list(data.values())])
                
            elif isinstance(data, list):
                # Validate multiple predictions
                for i, item in enumerate(data):
                    is_valid, validation_message = validate_clinical_data(item)
                    if not is_valid:
                        return jsonify({
                            'error': f"Validation error in item {i}: {validation_message}",
                            'prediction_id': prediction_id
                        }), 400
                
                X = np.array([list(d.values()) for d in data])
            else:
                return jsonify({
                    'error': 'Invalid JSON format.',
                    'prediction_id': prediction_id
                }), 400
                
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected.',
                    'prediction_id': prediction_id
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'error': 'Invalid file type. Only CSV and Excel files are allowed.',
                    'prediction_id': prediction_id
                }), 400
            
            # Save file securely
            filename = secure_filename(f"{prediction_id}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                # Validate CSV structure
                missing_features = [f for f in feature_names if f not in df.columns]
                if missing_features:
                    return jsonify({
                        'error': f"Missing required columns: {missing_features}",
                        'prediction_id': prediction_id
                    }), 400
                
                X = df[feature_names].values
                
            except Exception as e:
                return jsonify({
                    'error': f'Error reading file: {str(e)}',
                    'prediction_id': prediction_id
                }), 400
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            return jsonify({
                'error': 'No valid input provided. Send JSON or file.',
                'prediction_id': prediction_id
            }), 400

        # Make predictions
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = np.max(prob)
            prediction_class = class_names[pred]
            
            # Determine confidence level
            if confidence >= confidence_thresholds['high_confidence']:
                confidence_level = 'High'
                recommendation = 'Strong recommendation for follow-up'
            elif confidence >= confidence_thresholds['medium_confidence']:
                confidence_level = 'Medium'
                recommendation = 'Moderate recommendation for follow-up'
            else:
                confidence_level = 'Low'
                recommendation = 'Consider additional testing'
            
            # Calculate uncertainty
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            max_entropy = -np.log(0.5)
            uncertainty = entropy / max_entropy
            
            result = {
                'prediction': prediction_class,
                'confidence': float(confidence),
                'confidence_level': confidence_level,
                'uncertainty': float(uncertainty),
                'recommendation': recommendation,
                'probabilities': {
                    'benign': float(prob[0]),
                    'malignant': float(prob[1])
                },
                'risk_factors': {
                    'high_risk': prediction_class == 'Malignant' and confidence > 0.8,
                    'moderate_risk': prediction_class == 'Malignant' and 0.6 <= confidence <= 0.8,
                    'low_risk': prediction_class == 'Benign' or confidence < 0.6
                }
            }
            results.append(result)
        
        # Log prediction
        log_prediction(prediction_id, data if 'data' in locals() else 'file_upload', results, user_info)
        
        return jsonify({
            'prediction_id': prediction_id,
            'timestamp': datetime.utcnow().isoformat(),
            'results': results,
            'disclaimer': 'This prediction is for clinical decision support only. Final diagnosis should be made by qualified healthcare professionals.'
        })
        
    except Exception as e:
        logger.error(f"Prediction error for {prediction_id}: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred during prediction.',
            'prediction_id': prediction_id
        }), 500

@app.route('/predictions/<prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    """Retrieve a specific prediction by ID"""
    try:
        # In production, this would query a database
        # For demo, we'll read from the log file
        if os.path.exists('prediction_log.jsonl'):
            with open('prediction_log.jsonl', 'r') as f:
                for line in f:
                    log_entry = json.loads(line)
                    if log_entry['prediction_id'] == prediction_id:
                        return jsonify(log_entry)
        
        return jsonify({'error': 'Prediction not found'}), 404
        
    except Exception as e:
        logger.error(f"Error retrieving prediction {prediction_id}: {str(e)}")
        return jsonify({'error': 'Error retrieving prediction'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    logger.info("Starting Clinical Breast Cancer Prediction API")
    app.run(debug=False, host='0.0.0.0', port=5000)




