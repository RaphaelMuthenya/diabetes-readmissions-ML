
"""
Random Forest Readmission Prediction Model
Generated: 2025-08-21 15:22:01
"""

import joblib
import pandas as pd
import numpy as np
import json

class ReadmissionPredictor:
    def __init__(self, model_path, preprocessor_path, metadata_path):
        """Load saved model components"""
        
        # Load model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.categorical_features = self.metadata['feature_lists']['categorical_features']
        self.numeric_features = self.metadata['feature_lists']['numeric_features']
        
        print(f"[SUCCESS] Loaded {self.metadata['model_info']['model_type']} model")
        print(f"   Performance: {self.metadata['performance_metrics'].get('test_auc', 'N/A')} AUC")
        print(f"   Recall: {self.metadata['performance_metrics'].get('test_recall', 'N/A')}")
    
    def predict_readmission(self, patient_data):
        """
        Predict readmission probability for new patients
        
        Args:
            patient_data: pandas DataFrame with patient features
            
        Returns:
            dict with predictions and probabilities
        """
        
        # Validate input features
        expected_features = set(self.categorical_features + self.numeric_features)
        provided_features = set(patient_data.columns)
        
        missing_features = expected_features - provided_features
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features correctly
        patient_data_ordered = patient_data[self.categorical_features + self.numeric_features]
        
        # Preprocess data
        X_processed = self.preprocessor.transform(patient_data_ordered)
        
        # Make predictions
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        predictions = self.model.predict(X_processed)
        
        # Return results
        results = {
            'predictions': predictions.tolist(),
            'readmission_probabilities': probabilities.tolist(),
            'risk_categories': ['High' if p >= 0.5 else 'Low' for p in probabilities]
        }
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance rankings"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None

# Example usage:
# predictor = ReadmissionPredictor(
#     model_path='saved_models/readmission_rf_demo_20250821_152201.pkl',
#     preprocessor_path='saved_models/preprocessor_20250821_152201.pkl', 
#     metadata_path='saved_models/model_metadata_20250821_152201.json'
# )
# 
# results = predictor.predict_readmission(new_patient_data)
# print(results['readmission_probabilities'])
