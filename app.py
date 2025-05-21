import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

try:
    asthma_model = joblib.load('models/asthma_model.pkl')
    cancer_model = joblib.load('models/cancer_model.pkl')
    diabetes_model = joblib.load('models/diabetes_model.pkl')
    stroke_model = joblib.load('models/stroke_model.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure .pkl files are in the same directory.")
    st.stop()

# Initializing SHAP explainer for each model
# Using TreeExplainer since we're using XGBoost models
asthma_explainer = shap.TreeExplainer(asthma_model)
cancer_explainer = shap.TreeExplainer(cancer_model)
diabetes_explainer = shap.TreeExplainer(diabetes_model)
stroke_explainer = shap.TreeExplainer(stroke_model)

def get_ai_explanation(disease, model, explainer, input_data, feature_names):
    input_data_2d = input_data.reshape(1, -1)
    shap_values = explainer.shap_values(input_data_2d)
    shap_contributions = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    total_shap = np.abs(shap_contributions).sum()
    if total_shap > 1.5: 
        risk_level = "High Risk"
    elif total_shap > 0.5:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"
    feature_contributions = list(zip(feature_names, shap_contributions))
    feature_contributions = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
    explanation = []
    if disease == "Asthma":
        typical_age = 30
        explanation.append(f"Age ({input_data[-1]}): {'Higher' if input_data[-1] > typical_age else 'Lower'} than typical age ({typical_age}).")
        top_features = [f"{name} (Value: {input_data[i]}, Contribution: {contrib:.2f})" 
                        for i, (name, contrib) in enumerate(feature_contributions[:3])]
        explanation.append(f"Top contributing factors: {', '.join(top_features)}.")
    elif disease == "Cancer":
        typical_age = 40
        explanation.append(f"Age ({input_data[1]}): {'Higher' if input_data[1] > typical_age else 'Lower'} than typical age ({typical_age}).")
        top_features = [f"{name} (Value: {input_data[i]}, Contribution: {contrib:.2f})" 
                        for i, (name, contrib) in enumerate(feature_contributions[:3])]
        explanation.append(f"Top contributing factors: {', '.join(top_features)}.")
    elif disease == "Diabetes":
        typical_glucose = 100
        typical_bmi = 25
        explanation.append(f"Glucose ({input_data[1]}): {'Higher' if input_data[1] > typical_glucose else 'Lower'} than typical ({typical_glucose}).")
        explanation.append(f"BMI ({input_data[5]}): {'Higher' if input_data[5] > typical_bmi else 'Lower'} than typical ({typical_bmi}).")
        top_features = [f"{name} (Value: {input_data[i]}, Contribution: {contrib:.2f})" 
                        for i, (name, contrib) in enumerate(feature_contributions[:3])]
        explanation.append(f"Top contributing factors: {', '.join(top_features)}.")
    elif disease == "Stroke":
        typical_glucose = 100
        typical_bmi = 25
        explanation.append(f"Average Glucose Level ({input_data[6]}): {'Higher' if input_data[6] > typical_glucose else 'Lower'} than typical ({typical_glucose}).")
        explanation.append(f"BMI ({input_data[7]}): {'Higher' if input_data[7] > typical_bmi else 'Lower'} than typical ({typical_bmi}).")
        top_features = [f"{name} (Value: {input_data[i]}, Contribution: {contrib:.2f})" 
                        for i, (name, contrib) in enumerate(feature_contributions[:3])]
        explanation.append(f"Top contributing factors: {', '.join(top_features)}.")
    fig, ax = plt.subplots()
    features = [f[0] for f in feature_contributions]
    values = [f[1] for f in feature_contributions]
    colors = ['red' if v > 0 else 'blue' for v in values]
    ax.barh(features, values, color=colors)
    ax.set_xlabel("SHAP Value (Impact on Prediction)")
    ax.set_title("Feature Importance for Prediction")
    st.pyplot(fig)

    return risk_level, explanation

st.set_page_config(layout="centered")
st.title("Disease Prediction Model")
st.write("Select a disease to predict and enter the symptoms to get a diagnosis, risk level, AI explanations, and a visualization of feature importance.")

disease = st.selectbox("Select Disease", ["Asthma", "Cancer", "Diabetes", "Stroke"])

if disease == "Asthma":
    st.subheader("Asthma Prediction")
    with st.form("asthma_form"):
        tiredness = st.selectbox("Tiredness", [0, 1])
        dry_cough = st.selectbox("Dry Cough", [0, 1])
        difficulty_breathing = st.selectbox("Difficulty in Breathing", [0, 1])
        sore_throat = st.selectbox("Sore Throat", [0, 1])
        pains = st.selectbox("Pains", [0, 1])
        nasal_congestion = st.selectbox("Nasal Congestion", [0, 1])
        runny_nose = st.selectbox("Runny Nose", [0, 1])
        gender = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
        age = st.number_input("Age", min_value=9, max_value=60, value=21)
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = np.array([tiredness, dry_cough, difficulty_breathing, sore_throat, pains, nasal_congestion, runny_nose, gender, age])
            feature_names = ['Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'gender', 'age']
            prediction = asthma_model.predict([input_data])[0]
            st.success(f"**Prediction**: {'Asthma' if prediction == 1 else 'No Asthma'}")
            risk_level, explanation = get_ai_explanation("Asthma", asthma_model, asthma_explainer, input_data, feature_names)
            st.write(f"**Risk Level**: {risk_level}")
            st.write("**Explanation**:")
            for exp in explanation:
                st.info(f"{exp}")

elif disease == "Cancer":
    st.subheader("Cancer Prediction")
    with st.form("cancer_form"):
        gender = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
        age = st.number_input("Age", min_value=0, max_value=120, value=21)
        smoking = st.selectbox("Smoking", [0, 1])
        yellow_fingers = st.selectbox("Yellow Fingers", [0, 1])
        peer_pressure = st.selectbox("Peer Pressure", [0, 1])
        chronic_disease = st.selectbox("Chronic Disease", [0, 1])
        fatigue = st.selectbox("Fatigue", [0, 1])
        allergy = st.selectbox("Allergy", [0, 1])
        wheezing = st.selectbox("Wheezing", [0, 1])
        alcohol = st.selectbox("Alcohol Consuming", [0, 1])
        coughing = st.selectbox("Coughing", [0, 1])
        shortness_breath = st.selectbox("Shortness of Breath", [0, 1])
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", [0, 1])
        chest_pain = st.selectbox("Chest Pain", [0, 1])
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = np.array([gender, age, smoking, yellow_fingers, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, shortness_breath, swallowing_difficulty, chest_pain])
            feature_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
            prediction = cancer_model.predict([input_data])[0]
            st.success(f"**Prediction**: {'Cancer' if prediction == 1 else 'No Cancer'}")
            # AI Explanation with SHAP
            risk_level, explanation = get_ai_explanation("Cancer", cancer_model, cancer_explainer, input_data, feature_names)
            st.write(f"**Risk Level**: {risk_level}")
            st.write("**Explanation**:")
            for exp in explanation:
                st.info(f"{exp}")

elif disease == "Diabetes":
    st.subheader("Diabetes Prediction")
    with st.form("diabetes_form"):
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=6)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=148)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=250, value=72)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=150, value=35)
        insulin = st.number_input("Insulin", min_value=0, max_value=250, value=0)
        bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=33.6)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=1.5, value=0.627)
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            prediction = diabetes_model.predict([input_data])[0]
            st.success(f"**Prediction**: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
            risk_level, explanation = get_ai_explanation("Diabetes", diabetes_model, diabetes_explainer, input_data, feature_names)
            st.write(f"**Risk Level**: {risk_level}")
            st.write("**Explanation**:")
            for exp in explanation:
                st.info(f"{exp}")

elif disease == "Stroke":
    st.subheader("Stroke Prediction")
    with st.form("stroke_form"):
        gender = st.selectbox("Gender (0: Female, 1: Male, 2: Other)", [0, 1, 2])
        age = st.number_input("Age", min_value=0, max_value=120, value=67)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Ever Married", [0, 1])
        residence_type = st.selectbox("Residence Type (0: Rural, 1: Urban)", [0, 1])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=228.69)
        bmi = st.number_input("BMI", min_value=15.0, max_value=60.0, value=36.6)
        smoking_status = st.selectbox("Smoking Status (0: Never, 1: Formerly, 2: Smokes, 3: Unknown)", [0, 1, 2, 3])
        submitted = st.form_submit_button("Predict")
        if submitted:
            input_data = np.array([gender, age, hypertension, heart_disease, ever_married, residence_type, avg_glucose_level, bmi, smoking_status])
            feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
            prediction = stroke_model.predict([input_data])[0]
            st.success(f"**Prediction**: {'Stroke' if prediction == 1 else 'No Stroke'}")
            risk_level, explanation = get_ai_explanation("Stroke", stroke_model, stroke_explainer, input_data, feature_names)
            st.write(f"**Risk Level**: {risk_level}")
            st.write("**Explanation**:")
            for exp in explanation:
                st.info(f"{exp}")
