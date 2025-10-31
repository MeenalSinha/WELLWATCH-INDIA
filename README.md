# WellWatch India - Early Risk Detection for Lifestyle Diseases

## 🏥 System Overview
AI-powered early detection system for chronic lifestyle diseases in rural India.

## 📊 Model Performance
- **Accuracy**: 72.80%
- **Precision**: 0.7416
- **Recall**: 0.7280
- **F1 Score**: 0.7236
- **High-Risk Recall**: 79.21%

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

### 3. Or Use Google Colab
- Upload the entire package to Google Drive
- Open the notebook in Colab
- Run all cells sequentially

## 📁 Directory Structure
```
wellwatch_submission/
├── models/               # Trained ML models
│   ├── chronic_risk_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_list.json
├── data/                 # Datasets
│   ├── raw_screening_data.csv
│   ├── cleaned_data.csv
│   └── wellwatch.db
├── outputs/              # Visualizations and reports
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── dashboard_visualizations.png
│   └── ...
├── streamlit_app.py      # Web application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 Key Features
- ✅ 28 clinical and lifestyle features
- ✅ XGBoost classifier with 72.80% accuracy
- ✅ Real-time risk prediction
- ✅ Offline-capable mobile deployment
- ✅ Multi-language support
- ✅ PDF report generation
- ✅ Voice alerts for low-literacy users
- ✅ Dashboard for health managers

## 📱 Mobile App Integration
The system is designed for integration with mobile apps used by Community Health Workers (CHWs).

### API Endpoint Example
```python
import json
from predict import predict_risk

patient_data = {
    'age': 55,
    'gender': 'Male',
    'systolic_bp': 145,
    'fasting_glucose': 128,
    # ... other features
}

result = predict_risk(patient_data)
print(json.dumps(result, indent=2))
```

## 🔬 Model Details
- **Algorithm**: XGBoost Classifier
- **Features**: 28 (clinical vitals + lifestyle + family history)
- **Classes**: Low / Medium / High Risk
- **Training Data**: 1,600 samples
- **Test Data**: 400 samples
- **Cross-validation**: Stratified 5-fold

## 📈 Impact Metrics
- Early detection of high-risk individuals
- Reduced healthcare costs through prevention
- Empowered community health workers
- Data-driven resource allocation

## 👥 Team
Developed for AI4Bharat Hackathon 2025

## 📧 Contact
For questions or support, please contact: support@wellwatch.in

## 📄 License
This project is developed for educational and social impact purposes.
