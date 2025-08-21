"""
Random Forest Readmission Prediction API Simulation
Connects the saved model to a web dashboard for live demonstrations
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web dashboard

# Global variables for model components
model = None
preprocessor = None
metadata = None
feature_names = None

def load_model_components():
    """Load the saved Random Forest model and preprocessor"""
    global model, preprocessor, metadata, feature_names
    
    try:
        # Find the most recent model files
        model_dir = 'saved_models'
        
        # Get the latest model files (assuming timestamp naming)
        model_files = [f for f in os.listdir(model_dir) if f.startswith('readmission_rf_')]
        preprocessor_files = [f for f in os.listdir(model_dir) if f.startswith('preprocessor_')]
        metadata_files = [f for f in os.listdir(model_dir) if f.startswith('model_metadata_')]
        
        if not (model_files and preprocessor_files and metadata_files):
            raise FileNotFoundError("Model files not found. Please run the model saving pipeline first.")
        
        # Use the most recent files
        latest_model = sorted(model_files)[-1]
        latest_preprocessor = sorted(preprocessor_files)[-1]
        latest_metadata = sorted(metadata_files)[-1]
        
        # Load components
        model = joblib.load(os.path.join(model_dir, latest_model))
        preprocessor = joblib.load(os.path.join(model_dir, latest_preprocessor))
        
        with open(os.path.join(model_dir, latest_metadata), 'r') as f:
            metadata = json.load(f)
        
        # Extract feature names
        categorical_features = metadata['feature_lists']['categorical_features']
        numeric_features = metadata['feature_lists']['numeric_features']
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model: {latest_model}")
        print(f"   Preprocessor: {latest_preprocessor}")
        print(f"   Categorical features: {len(categorical_features)}")
        print(f"   Numeric features: {len(numeric_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def create_patient_dataframe(patient_data):
    """Convert patient input to properly formatted DataFrame"""
    
    # Get feature lists from metadata
    categorical_features = metadata['feature_lists']['categorical_features']
    numeric_features = metadata['feature_lists']['numeric_features']
    
    # Create DataFrame with all required features
    df = pd.DataFrame([patient_data])
    
    # Ensure all categorical features exist
    for cat_feature in categorical_features:
        if cat_feature not in df.columns:
            df[cat_feature] = 'Unknown'  # Default value
    
    # Ensure all numeric features exist
    for num_feature in numeric_features:
        if num_feature not in df.columns:
            df[num_feature] = 0  # Default value
    
    # Reorder columns to match training data
    df = df[categorical_features + numeric_features]
    
    return df

@app.route('/api/predict', methods=['POST'])
def predict_readmission():
    """API endpoint for readmission prediction"""
    
    try:
        # Get patient data from request
        patient_data = request.json
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Create properly formatted DataFrame
        patient_df = create_patient_dataframe(patient_data)
        
        # Preprocess the data
        X_processed = preprocessor.transform(patient_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_processed)[0]
        prediction_class = model.predict(X_processed)[0]
        
        # Get readmission probability (class 1)
        readmission_prob = float(prediction_proba[1])
        
        # Determine risk level
        if readmission_prob < 0.3:
            risk_level = 'Low'
            risk_color = '#2ecc71'
        elif readmission_prob < 0.6:
            risk_level = 'Medium' 
            risk_color = '#f39c12'
        else:
            risk_level = 'High'
            risk_color = '#e74c3c'
        
        # Generate recommendations based on risk level
        recommendations = generate_recommendations(readmission_prob, patient_data)
        
        # Get feature importance for this prediction
        if hasattr(model, 'feature_importances_'):
            # Get top 5 feature importances
            feature_importance = model.feature_importances_
            
            # Create feature names (simplified for demo)
            try:
                categorical_encoded = preprocessor.named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out(metadata['feature_lists']['categorical_features'])
                all_feature_names = list(categorical_encoded) + metadata['feature_lists']['numeric_features']
            except:
                all_feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
            
            # Get top features
            top_indices = np.argsort(feature_importance)[-5:][::-1]
            top_features = [
                {
                    'name': all_feature_names[i][:20] + '...' if len(all_feature_names[i]) > 20 else all_feature_names[i],
                    'importance': float(feature_importance[i])
                }
                for i in top_indices
            ]
        else:
            top_features = []
        
        # Return prediction results
        response = {
            'success': True,
            'prediction': {
                'readmission_probability': round(readmission_prob * 100, 1),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'prediction_class': int(prediction_class)
            },
            'recommendations': recommendations,
            'top_features': top_features,
            'model_info': {
                'model_type': metadata['model_info']['model_type'],
                'performance': {
                    'auc': metadata['performance_metrics'].get('test_auc', 'N/A'),
                    'recall': metadata['performance_metrics'].get('test_recall', 'N/A'),
                    'precision': metadata['performance_metrics'].get('test_precision', 'N/A')
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def generate_recommendations(risk_prob, patient_data):
    """Generate clinical recommendations based on risk probability and patient characteristics"""
    
    recommendations = []
    
    if risk_prob < 0.3:
        # Low risk recommendations
        recommendations = [
            "Standard discharge planning protocols",
            "Routine follow-up appointment in 1-2 weeks",
            "Patient education on medication compliance",
            "Emergency contact information provided"
        ]
    elif risk_prob < 0.6:
        # Medium risk recommendations
        recommendations = [
            "Enhanced discharge planning with care coordinator",
            "Follow-up appointment within 72 hours",
            "Medication reconciliation review",
            "Consider home health services",
            "Family/caregiver education session"
        ]
        
        # Add specific recommendations based on patient characteristics
        if patient_data.get('diabetes') in ['Yes', 'Yes_with_complications']:
            recommendations.append("Diabetes management education and monitoring")
        
        if int(patient_data.get('num_medications', 0)) > 15:
            recommendations.append("Pharmacy consultation for medication management")
            
    else:
        # High risk recommendations
        recommendations = [
            "🚨 PRIORITY: Intensive care management required",
            "Follow-up appointment within 24-48 hours",
            "Dedicated care coordinator assignment",
            "Transitional care program enrollment",
            "Medication adherence support program",
            "Family meeting for comprehensive discharge planning"
        ]
        
        # Add specific high-risk interventions
        if patient_data.get('discharge_disposition') == 'SNF':
            recommendations.append("SNF communication and care continuity plan")
        
        if int(patient_data.get('length_of_stay', 0)) > 7:
            recommendations.append("Extended stay protocol - assess for complications")
            
        if patient_data.get('admission_type') == 'Emergency':
            recommendations.append("Emergency admission follow-up - address underlying issues")
    
    return recommendations

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information and performance metrics"""
    
    if not metadata:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_info': metadata['model_info'],
        'performance_metrics': metadata['performance_metrics'],
        'training_info': metadata['training_info'],
        'feature_counts': {
            'categorical': len(metadata['feature_lists']['categorical_features']),
            'numeric': len(metadata['feature_lists']['numeric_features']),
            'total': len(metadata['feature_lists']['categorical_features']) + len(metadata['feature_lists']['numeric_features'])
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def dashboard():
    """Serve the updated dashboard with API integration"""
    
    dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Random Forest Readmission Predictor</title>
    <style>
        /* [Include the same CSS from the previous dashboard here] */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #4a6741 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .status-indicator {
            display: inline-block;
            padding: 5px 15px;
            background: rgba(46, 204, 113, 0.2);
            border-radius: 20px;
            margin-top: 10px;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }

        .patient-input {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            border: 2px solid #e9ecef;
        }

        .prediction-output {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            border: 2px solid #e9ecef;
        }

        .section-title {
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 1.1em;
        }

        .form-group select,
        .form-group input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e1e8ed;
            border-radius: 10px;
            font-size: 1em;
            transition: all 0.3s ease;
            background: white;
        }

        .form-group select:focus,
        .form-group input:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }

        .predict-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(52, 152, 219, 0.3);
        }

        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .risk-card {
            text-align: center;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }

        .risk-percentage {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .risk-label {
            font-size: 1.3em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .recommendations {
            background: white;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #3498db;
            margin-bottom: 20px;
        }

        .recommendations h4 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .recommendations ul {
            color: #34495e;
            line-height: 1.6;
        }

        .recommendations li {
            margin-bottom: 8px;
            padding-left: 10px;
        }

        .loading {
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Live Random Forest Predictor</h1>
            <p>Real-time readmission prediction using your trained model</p>
            <div class="status-indicator">
                <span id="modelStatus">🔄 Connecting to model...</span>
            </div>
        </div>

        <div class="main-content">
            <div class="patient-input">
                <h3 class="section-title">📋 Patient Information</h3>
                
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" min="18" max="120" value="65">
                </div>

                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select id="gender">
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="admission_type">Admission Type</label>
                    <select id="admission_type">
                        <option value="Emergency">Emergency</option>
                        <option value="Elective">Elective</option>
                        <option value="Urgent">Urgent</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="length_of_stay">Length of Stay (days)</label>
                    <input type="number" id="length_of_stay" min="1" max="30" value="4">
                </div>

                <div class="form-group">
                    <label for="num_procedures">Number of Procedures</label>
                    <input type="number" id="num_procedures" min="0" max="10" value="1">
                </div>

                <div class="form-group">
                    <label for="num_medications">Number of Medications</label>
                    <input type="number" id="num_medications" min="0" max="50" value="15">
                </div>

                <div class="form-group">
                    <label for="diabetes">Diabetes Status</label>
                    <select id="diabetes">
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="Yes_with_complications">Yes (with complications)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="discharge_disposition">Discharge Disposition</label>
                    <select id="discharge_disposition">
                        <option value="Home">Home</option>
                        <option value="SNF">Skilled Nursing Facility</option>
                        <option value="Home_Health">Home with Health Service</option>
                        <option value="Rehab">Rehabilitation Facility</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <button class="predict-btn" id="predictBtn" onclick="predictWithAPI()">
                    🔮 Predict with Random Forest Model
                </button>
            </div>

            <div class="prediction-output">
                <h3 class="section-title">🎯 Live Model Prediction</h3>
                
                <div id="predictionContent">
                    <div class="loading">
                        Click "Predict" to see live Random Forest results...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let modelLoaded = false;

        // Check model status on page load
        async function checkModelStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                const statusElement = document.getElementById('modelStatus');
                if (data.model_loaded) {
                    statusElement.innerHTML = '✅ Random Forest Model Active';
                    statusElement.style.background = 'rgba(46, 204, 113, 0.2)';
                    modelLoaded = true;
                } else {
                    statusElement.innerHTML = '❌ Model Not Loaded';
                    statusElement.style.background = 'rgba(231, 76, 60, 0.2)';
                }
            } catch (error) {
                document.getElementById('modelStatus').innerHTML = '⚠️ API Connection Error';
            }
        }

        async function predictWithAPI() {
            if (!modelLoaded) {
                alert('Model is not loaded. Please check the console for errors.');
                return;
            }

            const predictBtn = document.getElementById('predictBtn');
            const predictionContent = document.getElementById('predictionContent');
            
            // Disable button and show loading
            predictBtn.disabled = true;
            predictBtn.textContent = '🔄 Predicting...';
            predictionContent.innerHTML = '<div class="loading">Running Random Forest prediction...</div>';

            try {
                // Collect patient data
                const patientData = {
                    age: parseInt(document.getElementById('age').value),
                    gender: document.getElementById('gender').value,
                    admission_type: document.getElementById('admission_type').value,
                    length_of_stay: parseInt(document.getElementById('length_of_stay').value),
                    num_procedures: parseInt(document.getElementById('num_procedures').value),
                    num_medications: parseInt(document.getElementById('num_medications').value),
                    diabetes: document.getElementById('diabetes').value,
                    discharge_disposition: document.getElementById('discharge_disposition').value
                };

                // Make API call
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(patientData)
                });

                const result = await response.json();

                if (result.success) {
                    displayPredictionResult(result);
                } else {
                    displayError(result.error);
                }

            } catch (error) {
                displayError('Failed to connect to prediction API: ' + error.message);
            } finally {
                // Re-enable button
                predictBtn.disabled = false;
                predictBtn.textContent = '🔮 Predict with Random Forest Model';
            }
        }

        function displayPredictionResult(result) {
            const prediction = result.prediction;
            const recommendations = result.recommendations;
            const topFeatures = result.top_features;
            const modelInfo = result.model_info;

            let html = `
                <div class="risk-card" style="background: ${prediction.risk_color}; color: white;">
                    <div class="risk-percentage">${prediction.readmission_probability}%</div>
                    <div class="risk-label">${prediction.risk_level} Risk</div>
                </div>

                <div class="recommendations">
                    <h4>📋 Clinical Recommendations</h4>
                    <ul>
                        ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;

            if (topFeatures && topFeatures.length > 0) {
                html += `
                    <div class="recommendations">
                        <h4>🔍 Key Risk Factors (Live from Model)</h4>
                        ${topFeatures.map(feature => `
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span>${feature.name}</span>
                                <strong>${feature.importance.toFixed(3)}</strong>
                            </div>
                        `).join('')}
                    </div>
                `;
            }

            if (modelInfo && modelInfo.performance) {
                html += `
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px;">
                        <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">${modelInfo.performance.auc}</div>
                            <div style="color: #7f8c8d; font-size: 0.9em;">ROC-AUC</div>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">${(modelInfo.performance.recall * 100).toFixed(0)}%</div>
                            <div style="color: #7f8c8d; font-size: 0.9em;">Recall</div>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">${modelInfo.model_type}</div>
                            <div style="color: #7f8c8d; font-size: 0.9em;">Algorithm</div>
                        </div>
                    </div>
                `;
            }

            document.getElementById('predictionContent').innerHTML = html;
        }

        function displayError(errorMessage) {
            document.getElementById('predictionContent').innerHTML = `
                <div class="error">
                    <strong>Prediction Error:</strong><br>
                    ${errorMessage}
                </div>
            `;
        }

        // Initialize
        checkModelStatus();
    </script>
</body>
</html>
    '''
    
    return render_template_string(dashboard_html)

if __name__ == '__main__':
    print("🚀 Starting Random Forest Prediction API Server")
    print("="*60)
    
    # Load model components
    if load_model_components():
        print("\n🌐 Starting web server...")
        print("   Dashboard: http://localhost:5000")
        print("   API Health: http://localhost:5000/api/health")
        print("   Prediction API: POST http://localhost:5000/api/predict")
        print("\n💡 Use Ctrl+C to stop the server")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model. Please ensure:")
        print("   1. Model files exist in 'saved_models' directory")
        print("   2. Run the model saving pipeline first")
        print("   3. Check file permissions")