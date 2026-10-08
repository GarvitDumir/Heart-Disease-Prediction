# Heart Disease Prediction
## 🚀 Live Demo

[[Click here to try the Heart Disease Prediction App](YOUR_STREAMLIT_APP_URL)](https://heart-disease-prediction-garvit.streamlit.app/)

A Machine Learning project that predicts the presence of heart disease using patient health and clinical information.

## Machine Learning Model

* Algorithm: K-Nearest Neighbors (KNN)
* Data preprocessing: One-Hot Encoding
* Feature scaling: StandardScaler
* Model deployment: Streamlit

## Input Features

The model uses the following features:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak
* ST Slope

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn
* Google Colab

## Project Structure

```text
Heart-Disease-Prediction/
│
├── app/
│   └── app.py
│
├── model/
│   ├── knn_heart_disease_model.pkl
│   ├── scaler_heart_disease_model.pkl
│   └── cols_heart_disease_model.pkl
│
├── notebook/
│   └── Heart.ipynb
│
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/GarvitDumir/Heart-Disease-Prediction.git
```

### 2. Open the project

```bash
cd Heart-Disease-Prediction
```

### 3. Install required libraries

```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

### 4. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Disclaimer

This project is developed for educational and machine learning purposes only. It is not intended to provide medical diagnosis or replace professional medical advice.

## Author

**Garvit Dumir**

B.Tech CSE
NIT Arunachal Pradesh
