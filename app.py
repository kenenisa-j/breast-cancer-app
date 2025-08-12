from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model and scaler
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError('Model or scaler file not found. Please train and save them as model.pkl and scaler.pkl.')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load training data statistics for missing value imputation
# Note: In a real implementation, you'd save these during training
TRAINING_STATS_PATH = 'training_stats.pkl'
if os.path.exists(TRAINING_STATS_PATH):
    training_stats = joblib.load(TRAINING_STATS_PATH)
else:
    # Fallback: create dummy stats (in real implementation, save actual training stats)
    training_stats = {
        'feature_means': np.zeros(model.n_features_in_),
        'feature_names': [f'feature_{i}' for i in range(model.n_features_in_)]
    }

CLASS_MAP = {0: 'Benign', 1: 'Malignant'}

# Standard feature mapping for breast cancer dataset
FEATURE_MAPPING = {
    # Mean values
    'radius_mean': 0, 'texture_mean': 1, 'perimeter_mean': 2, 'area_mean': 3, 'smoothness_mean': 4,
    'compactness_mean': 5, 'concavity_mean': 6, 'concave_points_mean': 7, 'symmetry_mean': 8, 'fractal_dimension_mean': 9,
    
    # Standard error values
    'radius_se': 10, 'texture_se': 11, 'perimeter_se': 12, 'area_se': 13, 'smoothness_se': 14,
    'compactness_se': 15, 'concavity_se': 16, 'concave_points_se': 17, 'symmetry_se': 18, 'fractal_dimension_se': 19,
    
    # Worst values
    'radius_worst': 20, 'texture_worst': 21, 'perimeter_worst': 22, 'area_worst': 23, 'smoothness_worst': 24,
    'compactness_worst': 25, 'concavity_worst': 26, 'concave_points_worst': 27, 'symmetry_worst': 28, 'fractal_dimension_worst': 29,
    
    # Alternative naming variations
    'radius': 0, 'texture': 1, 'perimeter': 2, 'area': 3, 'smoothness': 4,
    'compactness': 5, 'concavity': 6, 'concave_points': 7, 'concave_pc': 7, 'symmetry': 8, 'fractal_dim': 9,
    'fractal_dimension': 9, 'perimeter_sarea_se': 12, 'perimeter_varea_worst': 22,
    'texture_wo': 21, 'compactnes': 5, 'concavity_s': 16, 'concavity_v': 26,
    'symmetry_s': 18, 'symmetry_v': 28, 'fractal_dimension_wors': 29
}

def map_descriptive_features_to_indices(data, input_type="JSON"):
    """Map descriptive feature names to numerical indices expected by the model"""
    if input_type == "JSON":
        if isinstance(data, dict):
            # Single prediction
            mapped_data = {}
            for feature_name, value in data.items():
                if feature_name in FEATURE_MAPPING:
                    mapped_data[f'feature_{FEATURE_MAPPING[feature_name]}'] = value
                elif feature_name.startswith('feature_'):
                    # Already in correct format
                    mapped_data[feature_name] = value
            return mapped_data
        elif isinstance(data, list):
            # Batch prediction
            mapped_batch = []
            for sample in data:
                mapped_sample = {}
                for feature_name, value in sample.items():
                    if feature_name in FEATURE_MAPPING:
                        mapped_sample[f'feature_{FEATURE_MAPPING[feature_name]}'] = value
                    elif feature_name.startswith('feature_'):
                        mapped_sample[feature_name] = value
                mapped_batch.append(mapped_sample)
            return mapped_batch
    return data

def map_csv_features_to_indices(df):
    """Map CSV column names to feature indices expected by the model"""
    mapped_df = pd.DataFrame()
    
    for col in df.columns:
        if col in FEATURE_MAPPING:
            # Map descriptive name to feature index
            feature_index = FEATURE_MAPPING[col]
            mapped_df[f'feature_{feature_index}'] = df[col]
        elif col.startswith('feature_'):
            # Already in correct format
            mapped_df[col] = df[col]
        elif col.lower() == 'diagnosis' or col.lower() == 'target':
            # Skip target column if present
            continue
        else:
            # Unknown column - skip it
            app.logger.warning(f"Unknown column '{col}' in CSV - skipping")
    
    return mapped_df

def get_feature_names():
    """Get feature names from training stats or generate default names"""
    if 'feature_names' in training_stats:
        return training_stats['feature_names']
    return [f'feature_{i}' for i in range(model.n_features_in_)]

def validate_and_clean_data(data, input_type="JSON"):
    """
    Validate and clean input data for prediction
    Returns: cleaned_data, error_message
    """
    feature_names = get_feature_names()
    required_features = set(feature_names)
    
    try:
        if input_type == "JSON":
            if isinstance(data, dict):
                # Single prediction
                input_features = set(data.keys())
                missing_features = required_features - input_features
                
                if missing_features:
                    error_msg = f"Missing required features: {list(missing_features)}. "
                    error_msg += f"Required features: {list(required_features)}. "
                    error_msg += "Example format: {'feature_0': 17.99, 'feature_1': 10.38, ...}"
                    return None, error_msg
                
                # Keep only required features in correct order
                cleaned_sample = [data.get(feature, np.nan) for feature in feature_names]
                # Wrap single sample in a list for consistent processing
                cleaned_data = [cleaned_sample]
                
            elif isinstance(data, list):
                # Batch prediction
                if not data:
                    return None, "Empty list provided. Please provide at least one prediction sample."
                
                cleaned_batch = []
                for i, sample in enumerate(data):
                    if not isinstance(sample, dict):
                        return None, f"Sample {i} is not a dictionary. Each sample must be a dictionary of features."
                    
                    input_features = set(sample.keys())
                    missing_features = required_features - input_features
                    
                    if missing_features:
                        error_msg = f"Sample {i} missing required features: {list(missing_features)}. "
                        error_msg += f"Required features: {list(required_features)}"
                        return None, error_msg
                    
                    # Keep only required features in correct order
                    cleaned_sample = [sample.get(feature, np.nan) for feature in feature_names]
                    cleaned_batch.append(cleaned_sample)
                
                cleaned_data = cleaned_batch
            else:
                return None, "Invalid JSON format. Expected dictionary or list of dictionaries."
        
        elif input_type == "CSV":
            # CSV data is already a DataFrame
            input_features = set(data.columns)
            missing_features = required_features - input_features
            
            if missing_features:
                error_msg = f"Missing required columns: {list(missing_features)}. "
                error_msg += f"Required columns: {list(required_features)}"
                return None, error_msg
            
            # Keep only required columns in correct order
            cleaned_data = data[feature_names].values.tolist()
        
        return cleaned_data, None
        
    except Exception as e:
        return None, f"Error processing data: {str(e)}"

def handle_missing_values(data):
    """Replace missing values with training data means"""
    # Get the number of features from the first sample
    num_features = len(data[0]) if data else 0
    feature_means = training_stats.get('feature_means', np.zeros(num_features))
    
    cleaned_data = []
    for sample in data:
        cleaned_sample = []
        for i, value in enumerate(sample):
            # Check for various missing value representations
            if pd.isna(value) or value == '' or str(value).upper() in ['N/A', 'NA', 'NULL', 'NAN']:
                cleaned_sample.append(feature_means[i])
            else:
                cleaned_sample.append(value)
        cleaned_data.append(cleaned_sample)
    
    return cleaned_data

def auto_unit_conversion(data):
    """Auto-convert units if values appear to be in wrong scale"""
    feature_names = get_feature_names()
    
    # Define length-based features (typically in mm, but might be in cm)
    length_features = [i for i, name in enumerate(feature_names) 
                      if any(keyword in name.lower() for keyword in ['radius', 'perimeter', 'area', 'diameter'])]
    
    converted_data = []
    for sample in data:
        converted_sample = []
        for i, value in enumerate(sample):
            if i in length_features and isinstance(value, (int, float)) and value < 10:
                # Likely in cm, convert to mm
                converted_sample.append(value * 10)
                logger.info(f"Auto-converted feature {feature_names[i]} from {value} to {value * 10} (cm to mm)")
            else:
                converted_sample.append(value)
        converted_data.append(converted_sample)
    
    return converted_data

def make_predictions(data, original_data=None):
    """Make predictions with confidence scores"""
    try:
        X = np.array(data, dtype=float)
        X_scaled = scaler.transform(X)
        
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        index_to_feature = {v:k for k,v in FEATURE_MAPPING.items()}
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = float(np.max(prob))
            result = {
                'sample_id': i,
                'prediction': CLASS_MAP.get(pred, str(pred)),
                'confidence': confidence,
                'probabilities': {
                    'benign': float(prob[0]),
                    'malignant': float(prob[1])
                }
            }
            
            if original_data is not None:
                # Convert numbered features back to named features
                feature_values = {}
                if isinstance(original_data, list):
                    sample_data = original_data[i]
                else:
                    sample_data = original_data
                    
                for feature_key, value in sample_data.items():
                    if feature_key.startswith('feature_'):
                        idx = int(feature_key.replace('feature_', ''))
                        if idx in index_to_feature:
                            feature_values[index_to_feature[idx]] = value
                    else:
                        feature_values[feature_key] = value
                
                result['feature_values'] = feature_values
            
            results.append(result)
        
        
        
        return results, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint for JSON input"""
    start_time = datetime.now()
    logger.info(f"Prediction request received at {start_time}")
    
    try:
        if not request.content_type or not request.content_type.startswith('application/json'):
            return jsonify({
                'error': 'Content-Type must be application/json',
                'example': {
                    'single_prediction': {'feature_0': 17.99, 'feature_1': 10.38},
                    'batch_prediction': [{'feature_0': 17.99, 'feature_1': 10.38}, {'feature_0': 20.57, 'feature_1': 17.77}]
                }
            }), 400
        
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        logger.info(f"Processing {len(data) if isinstance(data, list) else 1} prediction(s)")
        
        # Map descriptive feature names to indices if needed
        mapped_data = map_descriptive_features_to_indices(data, "JSON")
        
        # Validate and clean data
        cleaned_data, error = validate_and_clean_data(mapped_data, "JSON")
        if error:
            return jsonify({'error': error}), 400
        
        # Handle missing values
        cleaned_data = handle_missing_values(cleaned_data)
        logger.info("Missing values handled")
        
        # Auto unit conversion
        cleaned_data = auto_unit_conversion(cleaned_data)
        logger.info("Unit conversion applied")
        
        # Make predictions
        results, error = make_predictions(cleaned_data)
        if error:
            return jsonify({'error': error}), 500
        
        # Log results
        for result in results:
            logger.info(f"Sample {result['sample_id']}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Prediction completed in {processing_time:.3f} seconds")
        
        return jsonify({
            'results': results,
            'processing_time_seconds': processing_time,
            'samples_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Enhanced prediction endpoint for CSV file upload"""
    start_time = datetime.now()
    logger.info(f"CSV prediction request received at {start_time}")
    
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'example': 'Send CSV file with feature columns. Required columns: ' + str(get_feature_names())
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
            logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Map descriptive column names to feature indices
        mapped_df = map_csv_features_to_indices(df)
        logger.info(f"CSV columns mapped: {list(mapped_df.columns)}")
        
        # Validate and clean data
        cleaned_data, error = validate_and_clean_data(mapped_df, "CSV")
        if error:
            return jsonify({'error': error}), 400
        
        # Handle missing values
        cleaned_data = handle_missing_values(cleaned_data)
        logger.info("Missing values handled")
        
        # Auto unit conversion
        cleaned_data = auto_unit_conversion(cleaned_data)
        logger.info("Unit conversion applied")
        # Get the original data before mapping
        original_data = df.to_dict('records')
        # Make predictions
        results, error = make_predictions(cleaned_data, mapped_df.to_dict('records'))
        #results, error = make_predictions(cleaned_data)
        if error:
            return jsonify({'error': error}), 500
        
        # Log results summary
        predictions_summary = {}
        for result in results:
            pred = result['prediction']
            predictions_summary[pred] = predictions_summary.get(pred, 0) + 1
        
        logger.info(f"CSV predictions summary: {predictions_summary}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"CSV prediction completed in {processing_time:.3f} seconds")
        
        return jsonify({
            'results': results,
            'processing_time_seconds': processing_time,
            'samples_processed': len(results),
            'predictions_summary': predictions_summary
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in CSV prediction: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_count': model.n_features_in_ if model else 0,
        'feature_names': get_feature_names()
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': type(model).__name__,
        'feature_count': model.n_features_in_,
        'feature_names': get_feature_names(),
        'class_names': CLASS_MAP,
        'training_stats_available': 'feature_means' in training_stats
    })

if __name__ == '__main__':
    logger.info("Starting enhanced breast cancer prediction API")
    logger.info(f"Model loaded with {model.n_features_in_} features")
    app.run(debug=True)