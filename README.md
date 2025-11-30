# 🧠 Brain Stroke Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for predicting stroke risk based on patient health metrics and demographic information. This project demonstrates end-to-end data science workflows including data preprocessing, feature engineering, model training with cross-validation, and deployment via an interactive web application.

## 🌐 Live Demo

**Try the application here:** [Brain Stroke Predictor](https://brain-stroke-predector.streamlit.app/)

> Enter patient information and get instant stroke risk predictions powered by machine learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Pipeline](#model-pipeline)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Stroke is a leading cause of death and disability worldwide. Early identification of at-risk individuals can enable preventive interventions and reduce stroke incidence. This project leverages machine learning to predict stroke risk using patient demographic and clinical features.

The system employs a **Support Vector Machine (SVM)** classifier integrated within a scikit-learn pipeline that handles:
- Automatic preprocessing (scaling and encoding)
- Cross-validation for robust model evaluation
- Real-time predictions via a user-friendly Streamlit interface

---

## ✨ Features

- **End-to-End ML Pipeline**: Automated data preprocessing, feature engineering, model training, and evaluation
- **Interactive Web Application**: Built with Streamlit for real-time stroke risk assessment
- **Robust Model Performance**: Cross-validated SVM classifier with optimized hyperparameters
- **Production-Ready**: Serialized model pipeline using joblib for easy deployment
- **Comprehensive Documentation**: Detailed Jupyter notebooks documenting each phase of development
- **Class Imbalance Handling**: Implements class weighting to address dataset imbalance

---

## 📊 Dataset

The project uses the **Healthcare Dataset - Stroke Data** from Kaggle, containing patient records with the following features:

**Dataset Source**: [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

### Input Features
- **Demographic**: Gender, Age, Marital Status, Residence Type
- **Clinical**: Hypertension, Heart Disease, Average Glucose Level, BMI
- **Lifestyle**: Work Type, Smoking Status

### Target Variable
- **Stroke**: Binary classification (0 = No Stroke, 1 = Stroke)

**Dataset Characteristics**:
- Total Records: ~5,000 patients
- Class Distribution: Highly imbalanced (stroke cases are rare)
- Missing Values: Handled during preprocessing (BMI column)

---

## 📁 Project Structure

```
BrainStrokePredictor/
│
├── data_preprocessing.ipynb          # Data cleaning and feature engineering
├── model_training_evaluation.ipynb   # Model training, tuning, and evaluation
│
├── healthcare-dataset-stroke-data.csv        # Original dataset
├── healthcare-dataset-stroke-data_cleaned.csv # Preprocessed dataset
│
├── model.joblib                      # Serialized ML pipeline
├── prediction.py                     # Prediction logic module
├── app.py                           # Streamlit web application
│
└── README.md                        # Project documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sharjeel1042/BrainStrokePredictor.git
   cd BrainStrokePredictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Core Dependencies**:
   - `streamlit` - Web application framework
   - `pandas` - Data manipulation
   - `numpy` - Numerical computing
   - `scikit-learn` - Machine learning algorithms
   - `joblib` - Model serialization
   - `jupyter` - Notebook environment
   - `matplotlib` & `seaborn` - Data visualization

---

## 💻 Usage

### Running the Web Application

Launch the Streamlit app for interactive predictions:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Using the Prediction API

You can also use the prediction module programmatically:

```python
from prediction import predictFromuserInput
import pandas as pd

# Example patient data
patient_data = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

# Get prediction
prediction, probability = predictFromuserInput(patient_data)

print(f"Stroke Risk: {'High' if prediction[0] == 1 else 'Low'}")
print(f"Probability: {probability[0][1]*100:.2f}%")
```

### Exploring the Notebooks

1. **Data Preprocessing**: Open `data_preprocessing.ipynb` to see:
   - Exploratory data analysis
   - Missing value handling
   - Feature encoding strategies
   - Data cleaning process

2. **Model Training**: Open `model_training_evaluation.ipynb` to see:
   - Model selection and comparison
   - Hyperparameter tuning
   - Cross-validation results
   - Performance metrics

---

## 🔧 Model Pipeline

The prediction system uses a **scikit-learn Pipeline** that automatically handles all preprocessing steps:

```
Input Data
    ↓
Column Transformer
    ├─→ Numerical Features (age, avg_glucose_level, bmi, etc.)
    │   └─→ MinMaxScaler (normalize to [0, 1])
    │
    └─→ Categorical Features (gender, work_type, etc.)
        └─→ OneHotEncoder (convert to binary vectors)
    ↓
Support Vector Machine (SVM)
    • Kernel: RBF
    • Class Weighting: Balanced
    • Probability Estimates: Enabled
    ↓
Prediction + Probability Scores
```

### Key Pipeline Components

1. **Preprocessing**:
   - **MinMaxScaler**: Normalizes numerical features to [0, 1] range
   - **OneHotEncoder**: Converts categorical variables to binary vectors

2. **Classifier**:
   - **Algorithm**: Support Vector Machine (SVM)
   - **Kernel**: Radial Basis Function (RBF)
   - **Class Balancing**: Handles imbalanced dataset
   - **Cross-Validation**: 5-fold CV for robust evaluation

---

## 🛠 Technical Implementation

### Data Preprocessing

```python
# Key preprocessing steps
- Drop ID column (non-predictive)
- Handle missing BMI values (drop rows with nulls)
- Encode categorical features using OneHotEncoder
- Scale numerical features using MinMaxScaler
- Split data: 80% training, 20% testing
```

### Model Training

```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', column_transformer),
    ('classifier', SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    ))
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train final model
pipeline.fit(X_train, y_train)

# Save model
import joblib
joblib.dump(pipeline, 'model.joblib')
```

### Model Evaluation

The model is evaluated using multiple metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Breakdown of prediction outcomes

---

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| Cross-Validation Accuracy | ~94-95% |
| Test Set Accuracy | ~95% |
| F1-Score | Optimized for class imbalance |
| ROC-AUC | High discrimination capability |

**Note**: Given the highly imbalanced nature of stroke data, the model prioritizes recall (detecting actual stroke cases) while maintaining precision to minimize false alarms.

### Key Insights

- **Most Influential Features**: Age, hypertension, heart disease, average glucose level, and BMI
- **Class Imbalance**: Successfully handled using class weighting in SVM
- **Generalization**: Cross-validation confirms robust performance on unseen data

---

## 🔮 Future Enhancements

- [ ] **Advanced Models**: Implement ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] **Hyperparameter Tuning**: Grid search or Bayesian optimization for optimal parameters
- [ ] **Feature Engineering**: Create interaction features and polynomial features
- [ ] **SMOTE Integration**: Synthetic minority oversampling for better class balance
- [ ] **Model Interpretability**: Add SHAP or LIME for explainable predictions
- [ ] **API Development**: REST API using FastAPI for integration with other systems
- [ ] **Cloud Deployment**: Deploy to AWS/Azure/GCP for scalable production use
- [ ] **Monitoring Dashboard**: Track model performance and data drift over time
- [ ] **A/B Testing**: Compare multiple model versions in production

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sharjeel**

- GitHub: [@Sharjeel1042](https://github.com/Sharjeel1042)
- Project Link: [BrainStrokePredictor](https://github.com/Sharjeel1042/BrainStrokePredictor)

---

## 📚 References

- [World Health Organization - Stroke](https://www.who.int/news-room/fact-sheets/detail/stroke)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Support Vector Machines - Understanding the Math](https://scikit-learn.org/stable/modules/svm.html)

---

## ⚠️ Disclaimer

This is an educational project and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding a medical condition.

---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐️**

Made with ❤️ and Python

</div>
