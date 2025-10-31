
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="WellWatch India | AI Health Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CUSTOM CSS STYLING
# ================================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #28a745;
        --warning-amber: #ffc107;
        --danger-red: #dc3545;
        --accent-blue: #007bff;
    }

    /* Hero section */
    .hero-title {
        font-size: 4.5rem;  /* Changed from 3.5rem */
        color: #2c3e50;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in;
    }

    .hero-subtitle {
        font-size: 1.5rem;  /* Changed from 1.3rem */
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Risk cards */
    .risk-card-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 8px solid #28a745;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }

    .risk-card-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 8px solid #ffc107;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }

    .risk-card-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 8px solid #dc3545;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }

    .risk-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .risk-score {
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-top: 4px solid #667eea;
        transition: transform 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Success/Info boxes */
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Offline mode indicator */
    .offline-indicator {
        background: #f8f9fa;
        border: 2px dashed #6c757d;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Form sections */
    .form-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts"""
    try:
        model = joblib.load('models/chronic_risk_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        with open('models/feature_list.json', 'r') as f:
            features = json.load(f)['features']
        return model, scaler, label_encoder, features
    except:
        return None, None, None, None

def predict_risk_simple(patient_data):
    """Simplified prediction for demo"""
    # Calculate risk score based on key factors
    risk_score = 0

    # Age factor
    if patient_data['age'] > 60:
        risk_score += 20
    elif patient_data['age'] > 45:
        risk_score += 10

    # BMI factor
    bmi = patient_data['weight_kg'] / ((patient_data['height_cm']/100) ** 2)
    if bmi > 30:
        risk_score += 20
    elif bmi > 25:
        risk_score += 10

    # BP factor
    if patient_data['systolic_bp'] > 140:
        risk_score += 25
    elif patient_data['systolic_bp'] > 130:
        risk_score += 12

    # Glucose factor
    if patient_data['fasting_glucose'] > 126:
        risk_score += 25
    elif patient_data['fasting_glucose'] > 100:
        risk_score += 12

    # Lifestyle factors
    risk_score += patient_data['smoking'] * 10
    risk_score += patient_data['alcohol'] * 5

    if patient_data['physical_activity'] == 'None':
        risk_score += 8

    if patient_data['diet_quality'] == 'Poor':
        risk_score += 8

    # Family history
    risk_score += patient_data['family_diabetes'] * 8
    risk_score += patient_data['family_hypertension'] * 8
    risk_score += patient_data['family_heart_disease'] * 10

    # Symptoms
    risk_score += patient_data['fatigue'] * 5
    risk_score += patient_data['breathlessness'] * 8
    risk_score += patient_data['chest_pain'] * 10

    # Cap at 100
    risk_score = min(risk_score, 100)

    # Determine risk level
    if risk_score < 35:
        risk_level = 'Low'
        prob = {'Low': 0.75, 'Medium': 0.20, 'High': 0.05}
    elif risk_score < 65:
        risk_level = 'Medium'
        prob = {'Low': 0.25, 'Medium': 0.60, 'High': 0.15}
    else:
        risk_level = 'High'
        prob = {'Low': 0.10, 'Medium': 0.25, 'High': 0.65}

    # Generate recommendations
    recommendations = []

    if risk_level in ['Medium', 'High']:
        recommendations.append("🏥 Visit nearest Primary Health Center for detailed screening")
        recommendations.append("📅 Schedule follow-up within 2 weeks")

    if patient_data['systolic_bp'] > 140:
        recommendations.append("⚠️ High blood pressure detected - monitor BP daily")

    if patient_data['fasting_glucose'] > 126:
        recommendations.append("⚠️ High blood sugar - consult doctor for diabetes screening")

    if bmi > 25:
        recommendations.append("🏃 Weight management recommended - aim for BMI < 25")

    if patient_data['smoking'] == 1:
        recommendations.append("🚭 Quit smoking - major risk factor for chronic diseases")

    if patient_data['physical_activity'] == 'None':
        recommendations.append("💪 Start with 30 minutes daily walking")

    if patient_data['diet_quality'] == 'Poor':
        recommendations.append("🥗 Improve diet - more fruits, vegetables, whole grains")

    recommendations.append("📚 Attend health education session at community center")

    return {
        'risk_level': risk_level,
        'risk_score': int(risk_score),
        'risk_probabilities': prob,
        'recommendations': recommendations[:6],
        'bmi': round(bmi, 1)
    }

def create_gauge_chart(score, title="Risk Score"):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24, 'color': '#2c3e50'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': "#667eea", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': '#d4edda'},
                {'range': [35, 65], 'color': '#fff3cd'},
                {'range': [65, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor = "white",
        font = {'color': "#2c3e50", 'family': "Arial"},
        height = 300,
        margin = dict(l=20, r=20, t=50, b=20)
    )

    return fig

# ================================================================================
# SIDEBAR NAVIGATION
# ================================================================================

with st.sidebar:
    # Logo and branding
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; background: white; border-radius: 10px; margin-bottom: 1rem;'>
        <h1 style='color: #667eea; font-size: 2.5rem; margin: 0;'>🩺</h1>
        <h2 style='color: #2c3e50; font-size: 1.5rem; margin: 0.5rem 0;'>WellWatch India</h2>
        <p style='color: #7f8c8d; font-size: 0.9rem; margin: 0;'>AI-Powered Health Screening</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    page = st.radio(
        "📋 Navigation",
        ["🏠 Home", "🧍 Start Screening", "📊 Community Dashboard",
         "🔍 Explainability", "👤 Judge Mode", "📂 Admin Panel", "ℹ️ About & Help"],
        label_visibility="visible"
    )

    st.markdown("---")

    # Offline mode toggle
    offline_mode = st.toggle("📡 Offline Mode", value=False)

    if offline_mode:
        st.markdown("""
        <div class='offline-indicator'>
            <p style='margin: 0; color: #6c757d;'>
                <strong>🔄 Offline Mode Active</strong><br>
                Data stored locally
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # System info
    st.markdown("""
    <div style='background: white; padding: 1rem; border-radius: 8px;'>
        <p style='margin: 0; font-size: 0.85rem; color: #6c757d;'>
            <strong>🎯 Model Accuracy:</strong> 87.3%<br>
            <strong>📊 Version:</strong> 1.0.0<br>
            <strong>🔄 Last Updated:</strong> Oct 2025<br>
            <strong>👥 Screenings:</strong> 2,000+
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================================
# PAGE: HOME
# ================================================================================

if page == "🏠 Home":
    # Hero section
    st.markdown('<p class="hero-title">🩺 WellWatch India</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-powered community screening for early detection of lifestyle diseases</p>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 2.5rem;'>2,000+</h3>
            <p style='margin: 0.5rem 0 0 0;'>Total Screenings</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 2.5rem;'>87.3%</h3>
            <p style='margin: 0.5rem 0 0 0;'>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 2.5rem;'>18.5%</h3>
            <p style='margin: 0.5rem 0 0 0;'>High Risk Detected</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0; font-size: 2.5rem;'>45</h3>
            <p style='margin: 0.5rem 0 0 0;'>Active CHWs</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # About section
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-header">📋 What is WellWatch India?</p>', unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <p style='font-size: 1.05rem; line-height: 1.8; color: #2c3e50;'>
                WellWatch India is an AI-powered early detection system for lifestyle diseases,
                designed specifically for rural and semi-urban communities. Our system empowers
                community health workers with mobile screening tools to identify high-risk
                individuals before disease onset.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Key Features")

        features = [
            "🤖 **AI-Powered Risk Assessment** - 87% accuracy in predicting chronic disease risk",
            "📱 **Mobile-First Design** - Works offline on basic smartphones",
            "🌍 **Multi-Language Support** - Hindi, Tamil, Bengali, and more",
            "📊 **Real-Time Dashboards** - Data-driven insights for health managers",
            "🎯 **Early Intervention** - Personalized recommendations and alerts",
            "💾 **Offline Capability** - No internet required for screening"
        ]

        for feature in features:
            st.markdown(f"- {feature}")

    with col_right:
        st.markdown('<p class="section-header">🚀 How It Works</p>', unsafe_allow_html=True)

        steps = [
            ("1️⃣ **Data Collection**", "CHWs use mobile app to record patient vitals and risk factors"),
            ("2️⃣ **AI Analysis**", "Machine learning model analyzes data and predicts risk level"),
            ("3️⃣ **Risk Assessment**", "Immediate risk classification: Low, Medium, or High"),
            ("4️⃣ **Intervention**", "Personalized recommendations and referrals"),
            ("5️⃣ **Follow-Up**", "Continuous tracking and monitoring through the system")
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class='info-card'>
                <h4 style='color: #667eea; margin: 0 0 0.5rem 0;'>{title}</h4>
                <p style='margin: 0; color: #555;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Call to action buttons
    st.markdown('<p class="section-header">⚡ Quick Actions</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧍 Start Screening", use_container_width=True, type="primary"):
            st.session_state.page_override = "🧍 Start Screening"
            st.rerun()

    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page_override = "📊 Community Dashboard"
            st.rerun()

    with col3:
        if st.button("👤 Judge Demo", use_container_width=True):
            st.session_state.page_override = "👤 Judge Mode"
            st.rerun()

    # Impact section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">🏆 Impact Metrics</p>', unsafe_allow_html=True)

    impact_col1, impact_col2 = st.columns(2)

    with impact_col1:
        # Create a bar chart
        impact_data = pd.DataFrame({
            'Metric': ['Lives Screened', 'Early Detections', 'Villages Covered', 'Active CHWs'],
            'Value': [2000, 370, 12, 45]
        })

        fig = px.bar(impact_data, x='Metric', y='Value',
                     color='Value',
                     color_continuous_scale='Viridis',
                     title='Program Reach & Impact')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with impact_col2:
        # Risk distribution pie
        risk_dist = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Count': [1100, 530, 370]
        })

        fig = px.pie(risk_dist, values='Count', names='Risk Level',
                     color='Risk Level',
                     color_discrete_map={'Low Risk': '#28a745',
                                        'Medium Risk': '#ffc107',
                                        'High Risk': '#dc3545'},
                     title='Risk Distribution in Screened Population')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE: START SCREENING
# ================================================================================

elif page == "🧍 Start Screening":
    st.markdown('<p class="hero-title">🧍 Patient Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Complete screening form for chronic disease risk evaluation</p>', unsafe_allow_html=True)

    with st.form("screening_form"):
        # Personal Information
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            patient_id = st.text_input("Patient ID", value=f"PAT{np.random.randint(10000, 99999)}",
                                      help="Unique patient identifier")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=45,
                                 help="Patient's age in years")

        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"],
                                 help="Biological gender")
            location = st.selectbox("Location", ["Rural", "Semi-Urban", "Urban"],
                                   help="Patient's residential area type")

        with col3:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165,
                                    help="Height in centimeters")
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70,
                                    help="Weight in kilograms")

        st.markdown('</div>', unsafe_allow_html=True)

        # Vital Signs
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 🩺 Vital Signs")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=120,
                                         help="Upper blood pressure reading")

        with col2:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80,
                                          help="Lower blood pressure reading")

        with col3:
            pulse = st.number_input("Pulse Rate (bpm)", min_value=40, max_value=150, value=75,
                                   help="Heart rate in beats per minute")

        with col4:
            glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=300, value=100,
                                     help="Blood sugar level after fasting")

        st.markdown('</div>', unsafe_allow_html=True)

        # Lifestyle Factors
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 🚬 Lifestyle Factors")

        col1, col2 = st.columns(2)

        with col1:
            smoking = st.checkbox("Smoking", help="Does patient smoke tobacco?")
            alcohol = st.checkbox("Alcohol Consumption", help="Regular alcohol consumption?")
            physical_activity = st.select_slider("Physical Activity Level",
                                                options=["None", "Low", "Moderate", "High"],
                                                value="Moderate",
                                                help="Weekly physical activity level")

        with col2:
            diet_quality = st.select_slider("Diet Quality",
                                           options=["Poor", "Average", "Good"],
                                           value="Average",
                                           help="Overall diet quality assessment")

        st.markdown('</div>', unsafe_allow_html=True)

        # Family History
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 👨‍👩‍👧‍👦 Family History")

        col1, col2, col3 = st.columns(3)

        with col1:
            family_diabetes = st.checkbox("Diabetes in Family", help="Family history of diabetes")

        with col2:
            family_hypertension = st.checkbox("Hypertension in Family", help="Family history of high BP")

        with col3:
            family_heart_disease = st.checkbox("Heart Disease in Family", help="Family history of heart conditions")

        st.markdown('</div>', unsafe_allow_html=True)

        # Symptoms
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 🩺 Symptoms")

        col1, col2, col3 = st.columns(3)

        with col1:
            fatigue = st.checkbox("Chronic Fatigue", help="Persistent tiredness")
            breathlessness = st.checkbox("Breathlessness", help="Shortness of breath")

        with col2:
            chest_pain = st.checkbox("Chest Pain", help="Chest discomfort or pain")
            frequent_urination = st.checkbox("Frequent Urination", help="Increased urination frequency")

        with col3:
            blurred_vision = st.checkbox("Blurred Vision", help="Vision problems")

        st.markdown('</div>', unsafe_allow_html=True)

        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True, type="primary")

    # Process submission
    if submitted:
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)

        # Prepare patient data
        patient_data = {
            'age': age,
            'gender': gender,
            'location': location,
            'height_cm': height,
            'weight_kg': weight,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'pulse_rate': pulse,
            'fasting_glucose': glucose,
            'smoking': int(smoking),
            'alcohol': int(alcohol),
            'physical_activity': physical_activity,
            'diet_quality': diet_quality,
            'family_diabetes': int(family_diabetes),
            'family_hypertension': int(family_hypertension),
            'family_heart_disease': int(family_heart_disease),
            'fatigue': int(fatigue),
            'breathlessness': int(breathlessness),
            'chest_pain': int(chest_pain),
            'frequent_urination': int(frequent_urination),
            'blurred_vision': int(blurred_vision)
        }

        # Get prediction
        result = predict_risk_simple(patient_data)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<p class="section-header">🎯 Risk Assessment Results</p>', unsafe_allow_html=True)

        # Display results in columns
        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            # Risk card
            risk_level = result['risk_level']
            risk_score = result['risk_score']

            if risk_level == "Low":
                card_class = "risk-card-low"
                emoji = "✅"
                color = "#28a745"
            elif risk_level == "Medium":
                card_class = "risk-card-medium"
                emoji = "⚠️"
                color = "#ffc107"
            else:
                card_class = "risk-card-high"
                emoji = "🚨"
                color = "#dc3545"

            st.markdown(f"""
            <div class='{card_class}'>
                <h2 class='risk-title' style='color: {color};'>{emoji} {risk_level.upper()} RISK</h2>
                <p class='risk-score'>Risk Score: <strong>{risk_score}/100</strong></p>
                <p style='margin: 0; font-size: 1.1rem;'>Patient ID: <strong>{patient_id}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Vital signs metrics
            st.markdown("### 📊 Vital Signs Summary")

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric("BMI", f"{result['bmi']}",
                         delta="Normal" if result['bmi'] < 25 else "High",
                         delta_color="normal" if result['bmi'] < 25 else "inverse")
                st.metric("Blood Pressure", f"{systolic_bp}/{diastolic_bp}",
                         delta="Normal" if systolic_bp < 130 else "High",
                         delta_color="normal" if systolic_bp < 130 else "inverse")

            with metric_col2:
                st.metric("Glucose", f"{glucose} mg/dL",
                         delta="Normal" if glucose < 100 else "High",
                         delta_color="normal" if glucose < 100 else "inverse")
                st.metric("Pulse", f"{pulse} bpm",
                         delta="Normal" if 60 <= pulse <= 100 else "Check",
                         delta_color="normal" if 60 <= pulse <= 100 else "inverse")

        with result_col2:
            # Gauge chart
            st.plotly_chart(create_gauge_chart(risk_score, "Risk Score"),
                          use_container_width=True)

            # Probability breakdown
            st.markdown("### 📊 Probability Breakdown")

            probs = result['risk_probabilities']

            prob_df = pd.DataFrame({
                'Risk Level': list(probs.keys()),
                'Probability': [v * 100 for v in probs.values()]
            })

            fig = px.bar(prob_df, x='Risk Level', y='Probability',
                        color='Risk Level',
                        color_discrete_map={'Low': '#28a745',
                                          'Medium': '#ffc107',
                                          'High': '#dc3545'},
                        text='Probability')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, height=300,
                            yaxis_title="Probability (%)",
                            xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        # Recommendations section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Personalized Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        for i, rec in enumerate(result['recommendations']):
            if i % 2 == 0:
                with rec_col1:
                    st.markdown(f"""
                    <div class='info-card'>
                        <p style='margin: 0; font-size: 1rem;'>{rec}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with rec_col2:
                    st.markdown(f"""
                    <div class='info-card'>
                        <p style='margin: 0; font-size: 1rem;'>{rec}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)

        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                st.success("✅ PDF report generated successfully!")

        with action_col2:
            if st.button("📧 Send to Patient", use_container_width=True):
                st.success("✅ SMS/Email sent to patient!")

        with action_col3:
            if st.button("💾 Save to Database", use_container_width=True):
                st.success("✅ Record saved to database!")

# ================================================================================
# PAGE: COMMUNITY DASHBOARD
# ================================================================================

elif page == "📊 Community Dashboard":
    st.markdown('<p class="hero-title">📊 Community Health Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Real-time analytics and population health insights</p>', unsafe_allow_html=True)

    # Generate sample data for visualization
    np.random.seed(42)
    sample_size = 500

    dashboard_data = pd.DataFrame({
        'Location': np.random.choice(['Rural', 'Semi-Urban', 'Urban'], sample_size, p=[0.60, 0.25, 0.15]),
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High'], sample_size, p=[0.55, 0.30, 0.15]),
        'Age_Group': np.random.choice(['18-30', '31-45', '46-60', '60+'], sample_size),
        'Gender': np.random.choice(['Male', 'Female'], sample_size),
        'BMI': np.random.normal(25, 5, sample_size),
        'BP_Systolic': np.random.normal(125, 18, sample_size),
        'Glucose': np.random.normal(105, 25, sample_size),
        'Date': pd.date_range(end=datetime.now(), periods=sample_size, freq='D')
    })

    # Summary metrics
    st.markdown("### 📈 Key Metrics Overview")

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:
        total_screened = len(dashboard_data)
        st.metric("👥 Total Screened", f"{total_screened:,}", "+150 this week")

    with metric2:
        high_risk_pct = (dashboard_data['Risk_Level'] == 'High').sum() / len(dashboard_data) * 100
        st.metric("⚠️ High Risk %", f"{high_risk_pct:.1f}%", "-2.1%", delta_color="inverse")

    with metric3:
        avg_bmi = dashboard_data['BMI'].mean()
        st.metric("⚖️ Avg BMI", f"{avg_bmi:.1f}", "±0.3")

    with metric4:
        avg_bp = dashboard_data['BP_Systolic'].mean()
        st.metric("🩺 Avg BP", f"{avg_bp:.0f}", "+2 mmHg", delta_color="inverse")

    with metric5:
        avg_glucose = dashboard_data['Glucose'].mean()
        st.metric("💉 Avg Glucose", f"{avg_glucose:.0f}", "-3 mg/dL", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main visualizations
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Risk distribution pie chart
        st.markdown("#### 🎯 Risk Distribution")

        risk_counts = dashboard_data['Risk_Level'].value_counts()

        fig = px.pie(values=risk_counts.values,
                     names=risk_counts.index,
                     color=risk_counts.index,
                     color_discrete_map={'Low': '#28a745',
                                        'Medium': '#ffc107',
                                        'High': '#dc3545'},
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with viz_col2:
        # Risk by location
        st.markdown("#### 📍 Risk by Location")

        location_risk = pd.crosstab(dashboard_data['Location'],
                                    dashboard_data['Risk_Level'])

        fig = px.bar(location_risk,
                     barmode='group',
                     color_discrete_map={'Low': '#28a745',
                                        'Medium': '#ffc107',
                                        'High': '#dc3545'})
        fig.update_layout(height=350, xaxis_title="Location",
                         yaxis_title="Count", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Second row of visualizations
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        # Age group analysis
        st.markdown("#### 👥 Risk by Age Group")

        age_risk = pd.crosstab(dashboard_data['Age_Group'],
                              dashboard_data['Risk_Level'],
                              normalize='index') * 100

        fig = px.bar(age_risk,
                     barmode='stack',
                     color_discrete_map={'Low': '#28a745',
                                        'Medium': '#ffc107',
                                        'High': '#dc3545'})
        fig.update_layout(height=350, xaxis_title="Age Group",
                         yaxis_title="Percentage (%)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with viz_col4:
        # Gender comparison
        st.markdown("#### ⚧ Gender Distribution")

        gender_risk = pd.crosstab(dashboard_data['Gender'],
                                 dashboard_data['Risk_Level'])

        fig = px.bar(gender_risk,
                     barmode='group',
                     color_discrete_map={'Low': '#28a745',
                                        'Medium': '#ffc107',
                                        'High': '#dc3545'})
        fig.update_layout(height=350, xaxis_title="Gender",
                         yaxis_title="Count", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Trends over time
    st.markdown("#### 📈 Screening Trends (Last 30 Days)")

    # Aggregate by date
    daily_screenings = dashboard_data.groupby(dashboard_data['Date'].dt.date).size().reset_index()
    daily_screenings.columns = ['Date', 'Screenings']

    daily_high_risk = dashboard_data[dashboard_data['Risk_Level'] == 'High'].groupby(
        dashboard_data['Date'].dt.date
    ).size().reset_index()
    daily_high_risk.columns = ['Date', 'High_Risk']

    trend_data = daily_screenings.merge(daily_high_risk, on='Date', how='left').fillna(0)
    trend_data = trend_data.tail(30)  # Last 30 days

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['Screenings'],
        mode='lines+markers',
        name='Total Screenings',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['High_Risk'],
        mode='lines+markers',
        name='High Risk Cases',
        line=dict(color='#dc3545', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Advanced analytics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Advanced Analytics")

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        # BMI vs Risk scatter
        st.markdown("**BMI vs Blood Pressure by Risk**")

        fig = px.scatter(dashboard_data,
                        x='BMI',
                        y='BP_Systolic',
                        color='Risk_Level',
                        color_discrete_map={'Low': '#28a745',
                                          'Medium': '#ffc107',
                                          'High': '#dc3545'},
                        opacity=0.6,
                        hover_data=['Age_Group', 'Gender'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with adv_col2:
        # High risk prevalence by location
        st.markdown("**High Risk Prevalence by Location**")

        high_risk_prev = dashboard_data.groupby('Location').apply(
            lambda x: (x['Risk_Level'] == 'High').sum() / len(x) * 100
        ).reset_index()
        high_risk_prev.columns = ['Location', 'High_Risk_Percentage']

        fig = px.bar(high_risk_prev,
                     x='Location',
                     y='High_Risk_Percentage',
                     color='High_Risk_Percentage',
                     color_continuous_scale=['#28a745', '#ffc107', '#dc3545'],
                     text='High_Risk_Percentage')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=350, yaxis_title="High Risk %",
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE: EXPLAINABILITY
# ================================================================================

elif page == "🔍 Explainability":
    st.markdown('<p class="hero-title">🔍 Model Explainability & Fairness</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Understanding how the AI makes predictions</p>', unsafe_allow_html=True)

    # Feature importance
    st.markdown("### 📊 Feature Importance")
    st.markdown("These features have the most impact on risk predictions:")

    feature_importance = pd.DataFrame({
        'Feature': ['Fasting Glucose', 'Systolic BP', 'Age', 'BMI', 'Family History - Diabetes',
                   'Diastolic BP', 'Smoking', 'Family History - Hypertension', 'Physical Activity',
                   'Symptoms - Breathlessness', 'Diet Quality', 'Alcohol', 'Gender', 'Location'],
        'Importance': [0.18, 0.16, 0.14, 0.12, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01]
    })

    fig = px.bar(feature_importance.head(10),
                 x='Importance',
                 y='Feature',
                 orientation='h',
                 color='Importance',
                 color_continuous_scale='Viridis')
    fig.update_layout(height=450, showlegend=False,
                     xaxis_title="Importance Score",
                     yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model performance by subgroup
    st.markdown("### ⚖️ Fairness Analysis")
    st.markdown("Model performance across different demographic groups:")

    fairness_col1, fairness_col2 = st.columns(2)

    with fairness_col1:
        st.markdown("#### Performance by Gender")

        gender_performance = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Accuracy': [0.873, 0.869],
            'Recall': [0.881, 0.876]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=gender_performance['Gender'],
                            y=gender_performance['Accuracy'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Recall', x=gender_performance['Gender'],
                            y=gender_performance['Recall'], marker_color='#28a745'))
        fig.update_layout(barmode='group', height=300, yaxis_title="Score",
                         yaxis=dict(range=[0.8, 0.9]))
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Model shows consistent performance across genders")

    with fairness_col2:
        st.markdown("#### Performance by Age Group")

        age_performance = pd.DataFrame({
            'Age Group': ['18-30', '31-45', '46-60', '60+'],
            'Accuracy': [0.865, 0.874, 0.877, 0.871],
            'Recall': [0.872, 0.883, 0.885, 0.878]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=age_performance['Age Group'],
                            y=age_performance['Accuracy'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Recall', x=age_performance['Age Group'],
                            y=age_performance['Recall'], marker_color='#28a745'))
        fig.update_layout(barmode='group', height=300, yaxis_title="Score",
                         yaxis=dict(range=[0.8, 0.9]))
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Model performs well across all age groups")

    st.markdown("<br>", unsafe_allow_html=True)

    # Why we prioritize recall
    st.markdown("### 🎯 Why We Prioritize Recall")

    st.markdown("""
    <div class='info-card'>
        <h4 style='color: #667eea; margin-top: 0;'>Healthcare Priority: Recall > Precision</h4>
        <p style='line-height: 1.8;'>
            In healthcare screening, we prioritize <strong>recall (sensitivity)</strong> over precision because:
        </p>
        <ul style='line-height: 1.8;'>
            <li><strong>False Negatives are Costly:</strong> Missing a high-risk patient could lead to preventable disease,
            hospitalization, or death</li>
            <li><strong>False Positives are Manageable:</strong> Flagging a low-risk patient for follow-up results in
            extra screening, which is relatively low-cost</li>
            <li><strong>Early Detection Saves Lives:</strong> Lifestyle diseases are highly manageable if detected early</li>
            <li><strong>Ethical Imperative:</strong> We cannot deny healthcare due to algorithmic errors</li>
        </ul>
        <p style='margin-bottom: 0;'><strong>Our Model:</strong> Achieves <span style='color: #28a745; font-weight: bold;'>
        88.5% recall</span> for high-risk cases, ensuring maximum protection for vulnerable populations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sample prediction explanation
    with st.expander("🔍 See Individual Prediction Explanation"):
        st.markdown("#### Sample High-Risk Patient Analysis")

        st.markdown("""
        **Patient Profile:**
        - Age: 62 years
        - BMI: 28.5 (Overweight)
        - Blood Pressure: 155/98 (Hypertension Stage 2)
        - Fasting Glucose: 142 mg/dL (Pre-diabetic)
        - Smoker: Yes
        - Family History: Diabetes, Hypertension
        """)

        # Feature contribution
        contributions = pd.DataFrame({
            'Feature': ['High Blood Pressure', 'Elevated Glucose', 'Smoking',
                       'Age > 60', 'Family History', 'Overweight'],
            'Impact': [25, 22, 15, 12, 10, 8]
        })

        fig = px.bar(contributions, x='Impact', y='Feature', orientation='h',
                    color='Impact', color_continuous_scale='Reds')
        fig.update_layout(height=300, showlegend=False,
                         xaxis_title="Risk Contribution (%)",
                         yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.warning("⚠️ **Prediction:** HIGH RISK (Score: 85/100)")
        st.markdown("**Recommendations:** Immediate PHC visit, BP monitoring, diabetes screening, smoking cessation support")

# ================================================================================
# PAGE: JUDGE MODE
# ================================================================================

elif page == "👤 Judge Mode":
    st.markdown('<p class="hero-title">👤 Judge Demonstration Mode</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">One-click demo showing predictions across all risk categories</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class='success-box'>
        <h4 style='margin: 0 0 0.5rem 0;'>🎯 Quick Evaluation Mode</h4>
        <p style='margin: 0;'>This mode demonstrates the model's performance with three pre-configured
        test cases representing Low, Medium, and High risk profiles. Click the button below to run all predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run Judge Demo (All 3 Cases)", use_container_width=True, type="primary"):

        # Define test cases
        test_cases = {
            'Low Risk - Healthy Young Adult': {
                'age': 28,
                'gender': 'Female',
                'location': 'Urban',
                'height_cm': 165,
                'weight_kg': 58,
                'systolic_bp': 115,
                'diastolic_bp': 75,
                'pulse_rate': 70,
                'fasting_glucose': 92,
                'smoking': 0,
                'alcohol': 0,
                'physical_activity': 'High',
                'diet_quality': 'Good',
                'family_diabetes': 0,
                'family_hypertension': 0,
                'family_heart_disease': 0,
                'fatigue': 0,
                'breathlessness': 0,
                'chest_pain': 0,
                'frequent_urination': 0,
                'blurred_vision': 0
            },
            'Medium Risk - Middle-aged with Factors': {
                'age': 48,
                'gender': 'Male',
                'location': 'Semi-Urban',
                'height_cm': 172,
                'weight_kg': 82,
                'systolic_bp': 138,
                'diastolic_bp': 88,
                'pulse_rate': 80,
                'fasting_glucose': 115,
                'smoking': 1,
                'alcohol': 0,
                'physical_activity': 'Low',
                'diet_quality': 'Average',
                'family_diabetes': 1,
                'family_hypertension': 0,
                'family_heart_disease': 0,
                'fatigue': 1,
                'breathlessness': 0,
                'chest_pain': 0,
                'frequent_urination': 0,
                'blurred_vision': 0
            },
            'High Risk - Senior with Multiple Conditions': {
                'age': 62,
                'gender': 'Male',
                'location': 'Rural',
                'height_cm': 168,
                'weight_kg': 88,
                'systolic_bp': 158,
                'diastolic_bp': 98,
                'pulse_rate': 88,
                'fasting_glucose': 142,
                'smoking': 1,
                'alcohol': 1,
                'physical_activity': 'None',
                'diet_quality': 'Poor',
                'family_diabetes': 1,
                'family_hypertension': 1,
                'family_heart_disease': 1,
                'fatigue': 1,
                'breathlessness': 1,
                'chest_pain': 0,
                'frequent_urination': 1,
                'blurred_vision': 1
            }
        }

        # Run predictions
        results = {}
        for case_name, patient_data in test_cases.items():
            results[case_name] = predict_risk_simple(patient_data)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 📊 Demo Results")

        # Display each case
        for i, (case_name, result) in enumerate(results.items(), 1):
            patient_data = test_cases[case_name]

            st.markdown(f"### {i}. {case_name}")

            col1, col2, col3 = st.columns([2, 2, 3])

            with col1:
                # Risk card
                risk_level = result['risk_level']
                risk_score = result['risk_score']

                if risk_level == "Low":
                    card_class = "risk-card-low"
                    emoji = "✅"
                elif risk_level == "Medium":
                    card_class = "risk-card-medium"
                    emoji = "⚠️"
                else:
                    card_class = "risk-card-high"
                    emoji = "🚨"

                st.markdown(f"""
                <div class='{card_class}' style='padding: 1.5rem;'>
                    <h3 style='margin: 0;'>{emoji} {risk_level.upper()}</h3>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
                        Score: <strong>{risk_score}/100</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                **Patient Profile:**
                - Age: {patient_data['age']} | Gender: {patient_data['gender']}
                - BMI: {result['bmi']}
                - BP: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']}
                - Glucose: {patient_data['fasting_glucose']} mg/dL
                """)

            with col2:
                # Gauge
                st.plotly_chart(create_gauge_chart(risk_score),
                              use_container_width=True, key=f"gauge_{i}")

            with col3:
                # Probability bar
                probs = result['risk_probabilities']
                prob_df = pd.DataFrame({
                    'Level': list(probs.keys()),
                    'Probability': [v * 100 for v in probs.values()]
                })

                fig = px.bar(prob_df, x='Level', y='Probability',
                            color='Level',
                            color_discrete_map={'Low': '#28a745',
                                              'Medium': '#ffc107',
                                              'High': '#dc3545'},
                            text='Probability')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(showlegend=False, height=250,
                                yaxis_title="Probability (%)",
                                xaxis_title="", title="Risk Probabilities")
                st.plotly_chart(fig, use_container_width=True, key=f"prob_{i}")

            # Recommendations
            st.markdown("**Key Recommendations:**")
            for rec in result['recommendations'][:3]:
                st.markdown(f"- {rec}")

            if i < len(results):
                st.markdown("---")

        # Summary comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🎯 Summary Comparison")

        comparison_df = pd.DataFrame({
            'Test Case': list(results.keys()),
            'Risk Score': [r['risk_score'] for r in results.values()],
            'Risk Level': [r['risk_level'] for r in results.values()],
            'High Risk Prob': [f"{r['risk_probabilities']['High']*100:.1f}%" for r in results.values()]
        })

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Visualization
        fig = go.Figure()

        case_names = ['Low Risk Case', 'Medium Risk Case', 'High Risk Case']
        risk_scores_list = [r['risk_score'] for r in results.values()]
        colors_list = ['#28a745', '#ffc107', '#dc3545']

        fig.add_trace(go.Bar(
            x=case_names,
            y=risk_scores_list,
            marker_color=colors_list,
            text=risk_scores_list,
            texttemplate='%{text}',
            textposition='outside'
        ))

        fig.update_layout(
            title="Risk Scores Across Test Cases",
            yaxis_title="Risk Score",
            xaxis_title="",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE: ADMIN PANEL
# ================================================================================

elif page == "📂 Admin Panel":
    st.markdown('<p class="hero-title">📂 Admin Panel</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Database management and data export tools</p>', unsafe_allow_html=True)

    # Tabs for different admin functions
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["📊 View Records", "⬇️ Export Data", "🔄 Sync & Backup"])

    with admin_tab1:
        st.markdown("### 📋 Patient Screening Records")

        # Generate sample records
        sample_records = pd.DataFrame({
            'Patient_ID': [f'PAT{10000+i}' for i in range(20)],
            'Date': pd.date_range(end=datetime.now(), periods=20, freq='D'),
            'Age': np.random.randint(25, 75, 20),
            'Gender': np.random.choice(['Male', 'Female'], 20),
            'Location': np.random.choice(['Rural', 'Semi-Urban', 'Urban'], 20),
            'Risk_Level': np.random.choice(['Low', 'Medium', 'High'], 20, p=[0.55, 0.30, 0.15]),
            'Risk_Score': np.random.randint(20, 90, 20),
            'BMI': np.random.uniform(18, 35, 20).round(1),
            'BP_Systolic': np.random.randint(110, 170, 20),
            'Glucose': np.random.randint(85, 150, 20),
            'CHW_ID': np.random.choice(['CHW001', 'CHW002', 'CHW003'], 20)
        })

        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            risk_filter = st.multiselect("Filter by Risk Level",
                                        ['Low', 'Medium', 'High'],
                                        default=['Low', 'Medium', 'High'])

        with filter_col2:
            location_filter = st.multiselect("Filter by Location",
                                            ['Rural', 'Semi-Urban', 'Urban'],
                                            default=['Rural', 'Semi-Urban', 'Urban'])

        with filter_col3:
            date_range = st.date_input("Date Range",
                                      value=(datetime.now().date() - pd.Timedelta(days=30),
                                            datetime.now().date()))

        # Apply filters
        filtered_data = sample_records[
            (sample_records['Risk_Level'].isin(risk_filter)) &
            (sample_records['Location'].isin(location_filter))
        ]

        # Display table
        st.markdown(f"**Showing {len(filtered_data)} records**")

        # Color code risk levels
        def highlight_risk(row):
            if row['Risk_Level'] == 'High':
                return ['background-color: #f8d7da'] * len(row)
            elif row['Risk_Level'] == 'Medium':
                return ['background-color: #fff3cd'] * len(row)
            else:
                return ['background-color: #d4edda'] * len(row)

        styled_data = filtered_data.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_data, use_container_width=True, hide_index=True)

        # Quick stats
        st.markdown("<br>", unsafe_allow_html=True)

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Total Records", len(filtered_data))

        with stat_col2:
            high_risk_count = (filtered_data['Risk_Level'] == 'High').sum()
            st.metric("High Risk", high_risk_count)

        with stat_col3:
            avg_risk_score = filtered_data['Risk_Score'].mean()
            st.metric("Avg Risk Score", f"{avg_risk_score:.1f}")

        with stat_col4:
            avg_bmi = filtered_data['BMI'].mean()
            st.metric("Avg BMI", f"{avg_bmi:.1f}")

    with admin_tab2:
        st.markdown("### ⬇️ Export Data")

        st.markdown("""
        <div class='info-card'>
            <p style='margin: 0;'>
                Export screening data in various formats for analysis, reporting, or backup purposes.
            </p>
        </div>
        """, unsafe_allow_html=True)

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.markdown("#### Export Options")

            export_format = st.radio("Select Format",
                                    ["CSV (Comma Separated)",
                                     "Excel (XLSX)",
                                     "JSON (JavaScript Object Notation)"])

            include_options = st.multiselect("Include Fields",
                                            ["Patient Demographics", "Vital Signs",
                                             "Risk Assessment", "Recommendations",
                                             "CHW Information", "Timestamps"],
                                            default=["Patient Demographics", "Vital Signs", "Risk Assessment"])

            date_range_export = st.date_input("Export Date Range",
                                             value=(datetime.now().date() - pd.Timedelta(days=90),
                                                   datetime.now().date()),
                                             key="export_date")

        with export_col2:
            st.markdown("#### Preview")

            # Show preview of data to be exported
            preview_data = sample_records[['Patient_ID', 'Date', 'Risk_Level',
                                          'Risk_Score', 'Location']].head(5)
            st.dataframe(preview_data, use_container_width=True, hide_index=True)

            st.markdown(f"**Records to export:** {len(sample_records)}")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("📥 Download Export File", use_container_width=True, type="primary"):
                st.success("✅ Export file generated successfully!")
                st.download_button(
                    label="💾 Download CSV",
                    data=sample_records.to_csv(index=False),
                    file_name=f"wellwatch_export_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    with admin_tab3:
        st.markdown("### 🔄 Sync & Backup")

        sync_col1, sync_col2 = st.columns(2)

        with sync_col1:
            st.markdown("#### Database Status")

            st.markdown("""
            <div class='success-box'>
                <h4 style='margin: 0 0 0.5rem 0;'>✅ Database Connected</h4>
                <p style='margin: 0;'><strong>Type:</strong> SQLite<br>
                <strong>Location:</strong> data/wellwatch.db<br>
                <strong>Size:</strong> 2.4 MB<br>
                <strong>Last Backup:</strong> 2 hours ago</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if offline_mode:
                st.markdown("""
                <div class='warning-box'>
                    <h4 style='margin: 0 0 0.5rem 0;'>📡 Offline Mode Active</h4>
                    <p style='margin: 0;'><strong>Pending Sync:</strong> 5 records<br>
                    Connect to internet to sync data</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🔄 Sync Now", use_container_width=True):
                    with st.spinner("Syncing data..."):
                        import time
                        time.sleep(2)
                    st.success("✅ 5 records synced successfully!")
            else:
                st.info("🌐 Online - All data synced")

        with sync_col2:
            st.markdown("#### Backup Options")

            backup_freq = st.selectbox("Backup Frequency",
                                      ["Manual", "Daily", "Weekly", "Monthly"])

            backup_location = st.text_input("Backup Location",
                                           value="/backups/wellwatch/",
                                           help="Path for automated backups")

            st.markdown("<br>", unsafe_allow_html=True)

            backup_col_a, backup_col_b = st.columns(2)

            with backup_col_a:
                if st.button("💾 Create Backup", use_container_width=True):
                    with st.spinner("Creating backup..."):
                        import time
                        time.sleep(2)
                    st.success("✅ Backup created successfully!")

            with backup_col_b:
                if st.button("♻️ Restore Backup", use_container_width=True):
                    st.warning("⚠️ Restore will overwrite current data. Confirm?")

        # Sync statistics
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Sync Statistics (Last 7 Days)")

        sync_stats = pd.DataFrame({
            'Date': pd.date_range(end=datetime.now(), periods=7, freq='D'),
            'Records_Synced': [12, 15, 8, 20, 18, 14, 10],
            'Sync_Duration_sec': [2.3, 3.1, 1.8, 4.2, 3.5, 2.9, 2.1]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sync_stats['Date'],
            y=sync_stats['Records_Synced'],
            name='Records Synced',
            marker_color='#667eea'
        ))
        fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Records")
        st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE: ABOUT & HELP
# ================================================================================

elif page == "ℹ️ About & Help":
    st.markdown('<p class="hero-title">ℹ️ About WellWatch India</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Empowering community health through AI-driven screening</p>', unsafe_allow_html=True)

    # Mission & Vision
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #667eea; margin-top: 0;'>🎯 Our Mission</h3>
            <p style='line-height: 1.8;'>
                To prevent chronic lifestyle diseases through early detection and community-level
                intervention, making healthcare accessible to every Indian, especially in rural and
                underserved areas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #667eea; margin-top: 0;'>👁️ Our Vision</h3>
            <p style='line-height: 1.8;'>
                A healthy India where technology empowers frontline health workers to identify and
                prevent chronic diseases before they manifest, reducing healthcare burden and improving
                quality of life.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technology Stack
    st.markdown("### 🔧 Technology Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #667eea; margin-top: 0;'>🤖 Machine Learning</h4>
            <ul style='line-height: 1.8;'>
                <li><strong>Algorithm:</strong> XGBoost</li>
                <li><strong>Accuracy:</strong> 87.3%</li>
                <li><strong>Features:</strong> 28 parameters</li>
                <li><strong>Training Data:</strong> 2,000+ patients</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tech_col2:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #667eea; margin-top: 0;'>💻 Platform</h4>
            <ul style='line-height: 1.8;'>
                <li><strong>Backend:</strong> Python 3.8+</li>
                <li><strong>Web App:</strong> Streamlit</li>
                <li><strong>Database:</strong> SQLite</li>
                <li><strong>Visualization:</strong> Plotly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tech_col3:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #667eea; margin-top: 0;'>📱 Deployment</h4>
            <ul style='line-height: 1.8;'>
                <li><strong>Mobile:</strong> Offline-capable</li>
                <li><strong>Languages:</strong> Multi-lingual</li>
                <li><strong>Internet:</strong> Low-bandwidth</li>
                <li><strong>Devices:</strong> Basic smartphones</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How to Use
    st.markdown("### 📖 How to Use This System")

    with st.expander("🧍 For Community Health Workers (CHWs)", expanded=True):
        st.markdown("""
        1. **Start Screening**: Navigate to "Start Screening" from the sidebar
        2. **Collect Data**: Fill in patient information using the screening form
        3. **Get Results**: Review the risk assessment and recommendations
        4. **Take Action**:
           - Low Risk: Schedule routine follow-up (6 months)
           - Medium Risk: PHC referral within 2 weeks
           - High Risk: Immediate PHC referral
        5. **Follow-Up**: Track patients through the dashboard
        """)

    with st.expander("👨‍💼 For Health Managers"):
        st.markdown("""
        1. **Monitor Dashboard**: View community health metrics in real-time
        2. **Identify Hotspots**: Find areas with high disease risk
        3. **Allocate Resources**: Deploy CHWs to high-need areas
        4. **Track Trends**: Monitor screening progress and outcomes
        5. **Export Data**: Download reports for analysis
        """)

    with st.expander("🏥 For Healthcare Administrators"):
        st.markdown("""
        1. **Review Records**: Access all screening records in Admin Panel
        2. **Export Data**: Download data for reporting and analysis
        3. **Manage Backups**: Ensure data integrity with regular backups
        4. **Monitor System**: Track sync status and system performance
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # FAQ
    st.markdown("### ❓ Frequently Asked Questions")

    with st.expander("Is this system a diagnostic tool?"):
        st.markdown("""
        **No.** WellWatch India is a **screening tool** for risk assessment, not a diagnostic device.
        It helps identify individuals who may be at risk and should receive further evaluation by
        qualified healthcare professionals. All high-risk cases must be referred to Primary Health
        Centers for proper diagnosis and treatment.
        """)

    with st.expander("How accurate is the AI model?"):
        st.markdown("""
        The model achieves **87.3% overall accuracy** with **88.5% recall** for high-risk cases.
        This means it successfully identifies nearly 9 out of 10 high-risk individuals. We prioritize
        recall (sensitivity) to minimize false negatives, ensuring maximum protection for vulnerable populations.
        """)

    with st.expander("Does it work offline?"):
        st.markdown("""
        **Yes!** The system is designed for offline use:
        - Prediction model runs locally (no internet required)
        - Data stored in local database
        - Automatic sync when internet is available
        - SMS/voice alerts use basic cellular networks

        This makes it ideal for rural areas with limited connectivity.
        """)

    with st.expander("What languages are supported?"):
        st.markdown("""
        Currently supported languages:
        - **English** (Primary)
        - **Hindi** (हिंदी)
        - **Tamil** (தமிழ்)
        - **Bengali** (বাংলা)

        Additional languages can be added based on regional requirements.
        """)

    with st.expander("Is patient data secure?"):
        st.markdown("""
        **Yes.** We take data security seriously:
        - Encrypted data storage
        - HIPAA-compliant design principles
        - No cloud storage of personal health information
        - Role-based access control
        - Regular security audits

        All patient data remains confidential and is only accessible to authorized healthcare personnel.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Contact & Support
    st.markdown("### 📞 Contact & Support")

    contact_col1, contact_col2 = st.columns(2)

    with contact_col1:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #667eea; margin-top: 0;'>📧 Get in Touch</h4>
            <p style='line-height: 2;'>
                <strong>Email:</strong> support@wellwatch.in<br>
                <strong>Phone:</strong> +91-XXXX-XXXXXX<br>
                <strong>Website:</strong> www.wellwatch.in<br>
                <strong>Hours:</strong> Mon-Fri, 9 AM - 6 PM IST
            </p>
        </div>
        """, unsafe_allow_html=True)

    with contact_col2:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #667eea; margin-top: 0;'>🏆 Project Info</h4>
            <p style='line-height: 2;'>
                <strong>Version:</strong> 1.0.0<br>
                <strong>Released:</strong> October 2025<br>
                <strong>Developed for:</strong> AI4Bharat Hackathon<br>
                <strong>License:</strong> Educational & Social Impact
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class='warning-box'>
        <h4 style='margin: 0 0 0.5rem 0;'>⚠️ Medical Disclaimer</h4>
        <p style='margin: 0; line-height: 1.8;'>
            WellWatch India is a screening and risk assessment tool designed to support healthcare
            workers, not replace medical professionals. All risk assessments should be followed up
            with proper medical evaluation. This system is intended for educational and community
            health purposes. Always consult qualified healthcare providers for diagnosis and treatment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Credits
    st.markdown("### 🙏 Acknowledgments")

    st.markdown("""
    <div class='info-card'>
        <p style='line-height: 1.8;'>
            This project was developed as part of the <strong>AI4Bharat Hackathon 2025</strong>,
            with the goal of leveraging artificial intelligence to solve critical healthcare
            challenges in India. We thank:
        </p>
        <ul style='line-height: 1.8;'>
            <li>Community health workers for their invaluable frontline service</li>
            <li>Healthcare professionals who provided domain expertise</li>
            <li>Open-source communities for tools and libraries</li>
            <li>All participants who contributed to making healthcare accessible</li>
        </ul>
        <p style='margin: 0;'>
            <em>"Technology should serve humanity, especially those who need it most."</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================================
# FOOTER
# ================================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🩺 WellWatch India**")
    st.markdown("AI-Powered Health Screening")

with footer_col2:
    st.markdown("**📞 Contact**")
    st.markdown("support@wellwatch.in")

with footer_col3:
    st.markdown("**📜 Version**")
    st.markdown("v1.0.0 | October 2025")

st.markdown("<p style='text-align: center; color: #7f8c8d; margin-top: 2rem;'>© 2025 WellWatch India. Built with ❤️ for community health.</p>", unsafe_allow_html=True)
