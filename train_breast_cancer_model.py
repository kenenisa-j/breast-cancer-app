import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration - easy to change CSV file path
CSV_FILE_PATH = 'breast_cancer_data.csv'

def load_and_preprocess_data(file_path):
    """Load data from CSV and preprocess it"""
    try:
        # Load the CSV file
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Check if 'diagnosis' column exists
        if 'diagnosis' not in df.columns:
            raise ValueError("CSV must contain a 'diagnosis' column")
        
        # Separate features and target
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        print(f"Loaded {len(df)} samples with {len(X.columns)} features")
        print(f"Feature columns: {list(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Encode target: 'B' (benign) -> 0, 'M' (malignant) -> 1
        y_encoded = y.map({'B': 0, 'M': 1})
        
        # Check for any unmapped values
        if y_encoded.isna().any():
            unique_values = y.unique()
            raise ValueError(f"Target column contains unexpected values: {unique_values}. Expected: 'B' or 'M'")
        
        return X.values, y_encoded.values, list(X.columns)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please ensure the CSV file exists in the current directory.")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

def main():
    print("=== BREAST CANCER MODEL TRAINING ===")
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data(CSV_FILE_PATH)
    
    if X is None:
        print("Failed to load data. Exiting.")
        return
    
    # Split into train and test sets (80/20)
    print("Splitting data into training (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForestClassifier
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = clf.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the trained model and scaler (maintaining compatibility with app.py)
    print("\nSaving model and scaler...")
    joblib.dump(clf, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save training statistics for missing value imputation
    training_stats = {
        'feature_means': X_train.mean(axis=0),
        'feature_names': feature_names
    }
    joblib.dump(training_stats, 'training_stats.pkl')
    
    print("✅ Model and scaler saved successfully as 'model.pkl' and 'scaler.pkl'")
    print("✅ Training statistics saved as 'training_stats.pkl'")
    print("✅ Compatible with existing app.py backend")

if __name__ == "__main__":
    main()