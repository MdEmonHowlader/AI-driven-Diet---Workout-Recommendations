import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("diabetes_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'diabetes_model.pkl' and 'diabetes_scaler.pkl' are in the same directory.")
        return None, None

def analyze_risk_factors(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    """
    Analyze individual risk factors and provide explanations
    """
    risk_factors = []
    protective_factors = []
    
    # Define healthy ranges and risk thresholds
    if glucose > 140:
        risk_factors.append(f"🍯 **High Glucose Level ({glucose} mg/dL)**: Glucose levels above 140 mg/dL indicate impaired glucose tolerance, a strong predictor of diabetes.")
    elif glucose > 126:
        risk_factors.append(f"🍯 **Elevated Glucose ({glucose} mg/dL)**: Glucose levels above 126 mg/dL suggest potential insulin resistance.")
    elif glucose < 100:
        protective_factors.append(f"🍯 **Normal Glucose ({glucose} mg/dL)**: Healthy glucose levels reduce diabetes risk significantly.")
    
    if bmi > 30:
        risk_factors.append(f"⚖️ **Obesity (BMI: {bmi})**: BMI above 30 significantly increases insulin resistance and diabetes risk.")
    elif bmi > 25:
        risk_factors.append(f"⚖️ **Overweight (BMI: {bmi})**: BMI above 25 moderately increases diabetes risk.")
    elif bmi >= 18.5 and bmi <= 24.9:
        protective_factors.append(f"⚖️ **Healthy Weight (BMI: {bmi})**: Normal BMI range helps maintain insulin sensitivity.")
    
    if age > 45:
        risk_factors.append(f"🎂 **Advanced Age ({age} years)**: Age above 45 increases diabetes risk due to decreased insulin sensitivity.")
    elif age < 35:
        protective_factors.append(f"🎂 **Young Age ({age} years)**: Younger age is associated with better insulin sensitivity.")
    
    if pregnancies > 4:
        risk_factors.append(f"🤱 **Multiple Pregnancies ({pregnancies})**: Multiple pregnancies can increase insulin resistance.")
    elif pregnancies == 0:
        protective_factors.append(f"🤱 **No Previous Pregnancies**: Lower risk factor for diabetes development.")
    
    if blood_pressure > 90:
        risk_factors.append(f"💓 **High Blood Pressure ({blood_pressure} mmHg)**: Hypertension often accompanies insulin resistance.")
    elif blood_pressure < 80:
        protective_factors.append(f"💓 **Normal Blood Pressure ({blood_pressure} mmHg)**: Healthy blood pressure supports overall metabolic health.")
    
    if insulin > 200:
        risk_factors.append(f"💉 **High Insulin ({insulin} μU/mL)**: Elevated insulin levels indicate insulin resistance.")
    elif insulin > 125:
        risk_factors.append(f"💉 **Elevated Insulin ({insulin} μU/mL)**: Moderately high insulin suggests developing insulin resistance.")
    elif insulin < 100:
        protective_factors.append(f"💉 **Normal Insulin ({insulin} μU/mL)**: Healthy insulin levels indicate good metabolic function.")
    
    if dpf > 1.0:
        risk_factors.append(f"🧬 **High Genetic Risk ({dpf:.2f})**: Strong family history significantly increases diabetes risk.")
    elif dpf > 0.5:
        risk_factors.append(f"🧬 **Moderate Genetic Risk ({dpf:.2f})**: Some family history of diabetes increases risk.")
    elif dpf < 0.3:
        protective_factors.append(f"🧬 **Low Genetic Risk ({dpf:.2f})**: Minimal family history reduces diabetes risk.")
    
    if skin_thickness > 35:
        risk_factors.append(f"📐 **Thick Skin Fold ({skin_thickness} mm)**: May indicate insulin resistance and metabolic issues.")
    elif skin_thickness < 20:
        protective_factors.append(f"📐 **Normal Skin Thickness ({skin_thickness} mm)**: Healthy skin fold thickness.")
    
    return risk_factors, protective_factors

def get_lifestyle_recommendations(prediction, risk_factors):
    """
    Provide personalized lifestyle recommendations based on risk factors
    """
    recommendations = []
    
    if prediction == 1:  # High risk
        recommendations.append("🏥 **Immediate Action**: Consult with an endocrinologist or diabetes specialist within 1-2 weeks")
        recommendations.append("🩸 **Monitoring**: Check blood glucose levels regularly and maintain a glucose log")
        recommendations.append("💊 **Medication**: Discuss potential preventive medications with your doctor")
    
    # Diet recommendations
    recommendations.append("🥗 **Diet Plan**: Follow a low-glycemic index diet with complex carbohydrates")
    recommendations.append("🍽️ **Portion Control**: Use smaller plates and practice mindful eating")
    recommendations.append("⏰ **Meal Timing**: Eat regular meals every 3-4 hours to maintain stable blood sugar")
    
    # Exercise recommendations
    recommendations.append("🏃 **Cardio Exercise**: 30 minutes of moderate activity 5 days per week")
    recommendations.append("💪 **Strength Training**: 2-3 sessions per week to improve insulin sensitivity")
    recommendations.append("🚶 **Daily Activity**: Take at least 8,000-10,000 steps daily")
    
    # Lifestyle modifications
    recommendations.append("😴 **Sleep Quality**: Maintain 7-8 hours of quality sleep nightly")
    recommendations.append("🧘 **Stress Management**: Practice meditation, yoga, or stress-reduction techniques")
    recommendations.append("💧 **Hydration**: Drink plenty of water and limit sugary beverages")
    
    return recommendations

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    """
    Predict diabetes and return probability
    """
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return None, None
    
    # Create input array with the 8 features
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Get prediction and probability
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)
    
    return prediction[0], probability[0]

def main():
    # Page configuration
    st.set_page_config(
        page_title="Diabetes Risk Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .main-title {
            color: white;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-subtitle {
            color: #f8f9fa;
            font-size: 1.2rem;
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .info-card {
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #667eea;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .feature-container {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        .result-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 5px 20px rgba(79, 172, 254, 0.3);
        }
        .result-warning {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 5px 20px rgba(250, 112, 154, 0.3);
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            border-top: 3px solid #667eea;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏥 AI Diabetes Risk Predictor</h1>
        <p class="main-subtitle">Advanced machine learning for instant health assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Information section
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#667eea; margin-top:0;">📊 Health Parameters Assessment</h3>
        <p style="margin-bottom:0; color:#6c757d;">Please enter your health information below for an AI-powered diabetes risk analysis. All fields are required for accurate prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main container for inputs
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    
    # Create three columns for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Personal Info")
        pregnancies = st.number_input(
            "🤱 Pregnancies", 
            min_value=0, 
            max_value=20, 
            value=0,
            help="Number of times pregnant (0 for males)"
        )
        
        age = st.number_input(
            "🎂 Age (years)", 
            min_value=18, 
            max_value=120, 
            value=30,
            help="Age in years"
        )
        
        bmi = st.number_input(
            "⚖️ BMI", 
            min_value=10.0, 
            max_value=70.0, 
            value=25.0,
            step=0.1,
            help="Body mass index (weight in kg/(height in m)²)"
        )
    
    with col2:
        st.markdown("### 🩸 Blood Tests")
        glucose = st.number_input(
            "🍯 Glucose (mg/dL)", 
            min_value=0, 
            max_value=300, 
            value=120,
            help="Plasma glucose concentration"
        )
        
        insulin = st.number_input(
            "💉 Insulin (μU/mL)", 
            min_value=0, 
            max_value=1000, 
            value=85,
            help="2-Hour serum insulin level"
        )
        
        dpf = st.number_input(
            "🧬 Diabetes Pedigree", 
            min_value=0.0, 
            max_value=3.0, 
            value=0.5,
            step=0.01,
            help="Genetic predisposition factor"
        )
    
    with col3:
        st.markdown("### 📏 Physical Measurements")
        blood_pressure = st.number_input(
            "💓 Blood Pressure (mmHg)", 
            min_value=0, 
            max_value=200, 
            value=80,
            help="Diastolic blood pressure"
        )
        
        skin_thickness = st.number_input(
            "📐 Skin Thickness (mm)", 
            min_value=0, 
            max_value=100, 
            value=20,
            help="Triceps skin fold thickness"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Analyze Diabetes Risk", type="primary", use_container_width=True):
            # Get prediction
            prediction, probability = predict_diabetes(
                pregnancies, glucose, blood_pressure, skin_thickness, 
                insulin, bmi, dpf, age
            )
            
            if prediction is not None and probability is not None:
                # Get risk factor analysis
                risk_factors, protective_factors = analyze_risk_factors(
                    pregnancies, glucose, blood_pressure, skin_thickness, 
                    insulin, bmi, dpf, age
                )
                
                # Get lifestyle recommendations
                recommendations = get_lifestyle_recommendations(prediction, risk_factors)
                
                # Calculate percentages
                no_diabetes_prob = probability[0] * 100
                diabetes_prob = probability[1] * 100
                
                # Display results with enhanced styling
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                if prediction == 1:
                    # High risk
                    st.markdown(f"""
                    <div class="result-warning">
                        <h2 style="margin:0; font-size:2rem;">⚠️ HIGH RISK DETECTED</h2>
                        <p style="font-size:1.2rem; margin:10px 0;">Diabetes Probability: <strong>{diabetes_prob:.1f}%</strong></p>
                        <p style="font-size:1rem; opacity:0.9;">Immediate consultation with healthcare professional recommended</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Low risk
                    st.markdown(f"""
                    <div class="result-success">
                        <h2 style="margin:0; font-size:2rem;">✅ LOW RISK</h2>
                        <p style="font-size:1.2rem; margin:10px 0;">Healthy Status: <strong>{no_diabetes_prob:.1f}%</strong></p>
                        <p style="font-size:1rem; opacity:0.9;">Keep maintaining your healthy lifestyle!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced metrics display
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📈 Detailed Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:#667eea; margin:0;">✅ Healthy</h3>
                        <h2 style="color:#28a745; margin:5px 0;">{no_diabetes_prob:.1f}%</h2>
                        <p style="color:#6c757d; margin:0; font-size:0.9rem;">Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:#667eea; margin:0;">⚠️ At Risk</h3>
                        <h2 style="color:#dc3545; margin:5px 0;">{diabetes_prob:.1f}%</h2>
                        <p style="color:#6c757d; margin:0; font-size:0.9rem;">Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    risk_level = "HIGH" if diabetes_prob > 50 else "MODERATE" if diabetes_prob > 30 else "LOW"
                    risk_color = "#dc3545" if diabetes_prob > 50 else "#ffc107" if diabetes_prob > 30 else "#28a745"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:#667eea; margin:0;">🎯 Risk Level</h3>
                        <h2 style="color:{risk_color}; margin:5px 0;">{risk_level}</h2>
                        <p style="color:#6c757d; margin:0; font-size:0.9rem;">Assessment</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress bar with custom styling
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Risk Visualization")
                progress_color = "#dc3545" if diabetes_prob > 50 else "#ffc107" if diabetes_prob > 30 else "#28a745"
                st.markdown(f"""
                <div style="background:#f8f9fa; padding:1rem; border-radius:10px; margin:1rem 0;">
                    <p style="margin:0; color:#6c757d;">Diabetes Risk Level</p>
                    <div style="background:#e9ecef; border-radius:10px; height:20px; margin:10px 0;">
                        <div style="background:{progress_color}; width:{diabetes_prob}%; height:100%; border-radius:10px; transition:width 0.5s ease;"></div>
                    </div>
                    <p style="margin:0; text-align:right; color:#495057; font-weight:600;">{diabetes_prob:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk Factors Analysis Section
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🔍 Why This Prediction? - Risk Factor Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if risk_factors:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); padding:1.5rem; border-radius:15px; border-left:5px solid #e53e3e;">
                            <h4 style="color:#e53e3e; margin-top:0;">⚠️ Contributing Risk Factors</h4>
                        """, unsafe_allow_html=True)
                        
                        for factor in risk_factors:
                            st.markdown(f"<p style='margin:0.5rem 0; color:#742a2a; line-height:1.4;'>{factor}</p>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); padding:1.5rem; border-radius:15px; border-left:5px solid #38a169;">
                            <h4 style="color:#38a169; margin-top:0;">✅ No Major Risk Factors</h4>
                            <p style="margin:0; color:#276749;">Your health parameters are within healthy ranges!</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if protective_factors:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); padding:1.5rem; border-radius:15px; border-left:5px solid #38a169;">
                            <h4 style="color:#38a169; margin-top:0;">✅ Protective Factors</h4>
                        """, unsafe_allow_html=True)
                        
                        for factor in protective_factors:
                            st.markdown(f"<p style='margin:0.5rem 0; color:#276749; line-height:1.4;'>{factor}</p>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%); padding:1.5rem; border-radius:15px; border-left:5px solid #dd6b20;">
                            <h4 style="color:#dd6b20; margin-top:0;">📋 Areas for Improvement</h4>
                            <p style="margin:0; color:#9c4221;">Consider focusing on healthy lifestyle habits to build protective factors.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Personalized Recommendations Section
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Personalized Health Recommendations")
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding:1.5rem; border-radius:15px; border-left:5px solid #4299e1;">
                    <h4 style="color:#4299e1; margin-top:0;">💡 Action Plan Based on Your Results</h4>
                """, unsafe_allow_html=True)
                
                for i, recommendation in enumerate(recommendations[:8]):  # Limit to first 8 recommendations
                    st.markdown(f"<p style='margin:0.5rem 0; color:#2d3748; line-height:1.5;'><strong>{i+1}.</strong> {recommendation}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Summary explanation
                st.markdown("<br>", unsafe_allow_html=True)
                if prediction == 1:
                    explanation = f"""
                    <div style="background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); padding:1.5rem; border-radius:15px; border-left:5px solid #e53e3e;">
                        <h4 style="color:#e53e3e; margin-top:0;">🔬 Scientific Explanation</h4>
                        <p style="margin:0; color:#742a2a; line-height:1.6;">
                            Based on your health parameters, our AI model detected <strong>{len(risk_factors)} significant risk factors</strong> 
                            that increase your likelihood of developing diabetes. The combination of these factors resulted in a 
                            <strong>{diabetes_prob:.1f}% probability</strong> of diabetes risk. This prediction is based on patterns 
                            learned from thousands of similar health profiles in medical datasets.
                        </p>
                    </div>
                    """
                else:
                    explanation = f"""
                    <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); padding:1.5rem; border-radius:15px; border-left:5px solid #38a169;">
                        <h4 style="color:#38a169; margin-top:0;">🔬 Scientific Explanation</h4>
                        <p style="margin:0; color:#276749; line-height:1.6;">
                            Great news! Your health parameters show <strong>{len(protective_factors)} protective factors</strong> 
                            and only <strong>{len(risk_factors)} risk factors</strong>. This combination results in a 
                            <strong>{no_diabetes_prob:.1f}% probability</strong> of maintaining healthy glucose metabolism. 
                            Your current health profile aligns with patterns associated with low diabetes risk.
                        </p>
                    </div>
                    """
                
                st.markdown(explanation, unsafe_allow_html=True)
    
    # Enhanced footer with additional information
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Health tips section
    st.markdown("""
    <div class="info-card">
        <h3 style="color:#667eea; margin-top:0;">💡 Health Tips for Diabetes Prevention</h3>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem;">
            <div>
                <h4 style="color:#495057; margin-bottom:0.5rem;">🥗 Nutrition</h4>
                <p style="color:#6c757d; margin:0;">• Balanced diet with low sugar<br>• Regular meal timing<br>• Portion control</p>
            </div>
            <div>
                <h4 style="color:#495057; margin-bottom:0.5rem;">🏃 Exercise</h4>
                <p style="color:#6c757d; margin:0;">• 150 mins moderate activity/week<br>• Regular physical activity<br>• Strength training</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding:1.5rem; border-radius:15px; margin-top:2rem; border-left:5px solid #ffc107;">
        <div style="text-align:center;">
            <h4 style="color:#495057; margin:0 0 10px 0;">⚠️ Important Medical Disclaimer</h4>
            <p style="color:#6c757d; margin:0; line-height:1.6;">
                This AI tool provides risk assessments for <strong>educational purposes only</strong>. 
                It cannot replace professional medical consultation, diagnosis, or treatment. 
                Always consult qualified healthcare professionals for medical decisions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Credits
    st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding:1rem; color:#6c757d;">
        <p style="margin:0; font-size:0.9rem;">
            🤖 Powered by Advanced Machine Learning • 
            🏥 Healthcare AI Technology • 
            📊 Data-Driven Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
