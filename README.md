# Disease Prediction Model

## Overview
This project implements a **Disease Prediction Model** to predict the likelihood of four diseases—Asthma, Cancer, Diabetes, and Stroke—based on user-provided symptoms and health metrics. The models are trained using the XGBoost algorithm on datasets sourced from Kaggle. The project includes a Streamlit web application that allows users to input their symptoms, receive predictions, assess risk levels, and get AI-driven explanations using SHAP for feature importance.

### Features
- **Prediction Models**: Four XGBoost models trained for Asthma, Cancer, Diabetes, and Stroke.
- **AI Explanations**: Uses SHAP to provide feature importance and detailed explanations for predictions and risk levels.
- **Visualizations**: Displays bar charts of feature importance for each prediction.
- **Streamlit App**: A user-friendly web interface to input symptoms and view results.
- **Separate Training Scripts**: Individual scripts to train and save models for each disease as `.pkl` files.

## Project Structure
- **Folders**:
  - `asthma/`: Contains the asthma dataset (`asthma.csv`).
  - `cancer/`: Contains the cancer dataset (`cancer.csv`).
  - `diabetes/`: Contains the diabetes dataset (`diabetes.csv`).
  - `stroke/`: Contains the stroke dataset (`stroke.csv`).
- **Files**:
  - `app.py`: The Streamlit web application for predictions and explanations.
  - `train_asthma_model.py`: Script to train and save the Asthma model.
  - `train_cancer_model.py`: Script to train and save the Cancer model.
  - `train_diabetes_model.py`: Script to train and save the Diabetes model.
  - `train_stroke_model.py`: Script to train and save the Stroke model.
  - `asthma_model.pkl`, `cancer_model.pkl`, `diabetes_model.pkl`, `stroke_model.pkl`: Trained model files.
  - `requirements.txt`: List of Python dependencies.
  - `README.md`: Project documentation.

## Datasets
The datasets were sourced from Kaggle and preprocessed for this project. Each dataset contains features specific to the disease and a binary target variable (0 or 1) indicating the presence of the disease. The features were one-hot encoded where necessary.

- **Asthma Dataset** (`asthma.csv`):
  - Features: Tiredness, Dry-Cough, Difficulty-in-Breathing, Sore-Throat, Pains, Nasal-Congestion, Runny-Nose, gender, age.
  - Target: Asthma (0 or 1).
- **Cancer Dataset** (`cancer.csv`):
  - Features: GENDER, AGE, SMOKING, YELLOW_FINGERS, PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN.
  - Target: Cancer (0 or 1).
- **Diabetes Dataset** (`diabetes.csv`):
  - Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
  - Target: Outcome (0 or 1).
- **Stroke Dataset** (`stroke.csv`):
  - Features: gender, age, hypertension, heart_disease, ever_married, Residence_type, avg_glucose_level, bmi, smoking_status.
  - Target: Stroke (0 or 1).

## Dependencies
The project requires the following Python libraries. Install them using the provided `requirements.txt`:
```
streamlit==1.38.0
numpy==1.26.4
joblib==1.4.2
pandas==2.2.2
xgboost==2.1.0
shap==0.46.0
matplotlib==3.9.2
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup and Usage

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/Avinash00725/DiseasePredictionModel.git
cd DiseasePredictionModel
```

### 2. Train the Models
The project includes separate scripts to train each model. Run the following commands to train and save the models as `.pkl` files:
```bash
python train_asthma_model.py
python train_cancer_model.py
python train_diabetes_model.py
python train_stroke_model.py
```
This will generate `asthma_model.pkl`, `cancer_model.pkl`, `diabetes_model.pkl`, and `stroke_model.pkl` in the root directory.

**Note**: Ensure the dataset files (`asthma.csv`, `cancer.csv`, `diabetes.csv`, `stroke.csv`) are in their respective folders and the target column names in the training scripts match your dataset. Update the scripts if necessary (e.g., specify the correct target column name).

### 3. Run the Streamlit App Locally
Launch the Streamlit app to test it locally:
```bash
streamlit run app.py
```
This will open the app in your default browser (typically at `http://localhost:8501`).

### 4. Use the App
1. Select a disease (Asthma, Cancer, Diabetes, or Stroke) from the dropdown menu.
2. Fill in the symptom form with the required inputs (e.g., for Diabetes: Pregnancies, Glucose, etc.).
3. Click the "Predict" button to get:
   - The prediction (e.g., "Diabetes" or "No Diabetes").
   - The risk level (Low, Moderate, High) based on SHAP values.
   - AI explanations comparing your inputs to typical thresholds.
   - A bar chart showing feature importance (SHAP values).

### Example Usage
For Diabetes:
- **Input**: Pregnancies=6, Glucose=148, Blood Pressure=72, Skin Thickness=35, Insulin=0, BMI=33.6, Diabetes Pedigree Function=0.627, Age=50
- **Output**:
  - Prediction: Diabetes/No Diabetes
  - Risk Level: High/Moderate/Low Risk
  - AI Explanation:
    - "Glucose (148): Higher than typical (100)."
    - "BMI (33.6): Higher than typical (25)."
    - "Top contributing factors: Glucose (Value: 148, Contribution: 0.72), BMI (Value: 33.6, Contribution: 0.45), Age (Value: 50, Contribution: 0.30)."
  - Visualization: Bar chart of feature importance.

## Deployment on Streamlit Community Cloud
To deploy the app on Streamlit Community Cloud:
1. Ensure all files (including `.pkl` files) are pushed to your GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.
3. Click "New app" and select your repository (`DiseasePredictionModel`).
4. Specify the main script (`app.py`) and click "Deploy".
5. Once deployed, share the app URL with others.

## Future Enhancements
- **Improved Thresholds**: Replace placeholder typical thresholds (e.g., typical glucose=100) with actual clinical values or dataset averages.
- **Enhanced Explanations**: Add more detailed comparisons, such as percentile rankings of user inputs relative to the dataset.
- **Additional Visualizations**: Include more charts, such as waterfall plots for SHAP values or scatter plots of user inputs vs. population data.

## Acknowledgments
- Datasets sourced from [Kaggle](https://www.kaggle.com).
- Built with [Streamlit](https://streamlit.io), [XGBoost](https://xgboost.readthedocs.io), and [SHAP](https://shap.readthedocs.io).

## Author
- **Avinash** ([GitHub](https://github.com/Avinash00725))
