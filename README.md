# Machine-Learning-project-Student-Performance-Predictor
End-to-end ML project predicting student exam scores using Scikit-learn, cross-validation, feature engineering, and model evaluation.
# Student Performance Predictor

An end-to-end Machine Learning project that predicts students' exam scores using academic, personal, and environmental factors.

The project covers data preprocessing, exploratory data analysis, model training, cross-validation, feature interpretation, and prediction on unseen data.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Development:** Jupyter Notebook
- **Model Serialization:** Joblib

## Dataset

**Student Performance Factors Dataset — Kaggle**

- 6,607 student records
- 19 input features
- Target variable: `Exam_Score`

The dataset includes attendance, study hours, previous scores, parental involvement, access to resources, motivation level, and other student characteristics.

## Project Workflow

1. **Exploratory Data Analysis:** Studied feature distributions, correlations, missing values, and categorical relationships.
2. **Data Preprocessing:** Handled missing values using median/mode imputation, applied One-Hot Encoding to categorical features, and standardized numerical features.
3. **Model Training:** Trained Linear Regression and Random Forest Regressor.
4. **Cross-Validation:** Used 5-fold cross-validation to compare model performance.
5. **Model Evaluation:** Evaluated predictions using MAE, RMSE, and R².
6. **Error Analysis:** Investigated large prediction errors and examined model limitations.
7. **Feature Interpretation:** Analyzed Linear Regression coefficients and feature relationships.
8. **Model Saving:** Saved the complete preprocessing and prediction pipeline using Joblib.

## Model Performance

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| Cross-Validation MAE | 0.507 | 1.173 |
| Test MAE | **0.4524** | 1.0792 |
| Test RMSE | **1.8044** | 2.1560 |
| Test R² | **0.7696** | 0.6712 |

**Selected Model: Linear Regression**

Linear Regression achieved better performance than the tested Random Forest configuration.


## Key Findings

- Attendance and hours studied showed strong relationships with exam scores relative to the other numerical features.
- Linear Regression outperformed Random Forest during cross-validation and final test evaluation.
- Most predictions were close to actual exam scores.
- Seven test observations had prediction errors greater than 10 marks, highlighting limitations for unusual records.

## Predicting on New Data

The trained pipeline supports predictions on raw CSV files without manually performing One-Hot Encoding or scaling.

```python
import joblib
import pandas as pd

model = joblib.load("models/student_model.pkl")

data = pd.read_csv("new_students.csv")

X_new = data[list(model.feature_names_in_)]

data["Predicted_Exam_Score"] = model.predict(X_new)

data.to_csv("student_predictions.csv", index=False)
```

The CSV must contain the original 19 input columns with compatible values.

## Future Improvements

- Experiment with additional regression algorithms.
- Investigate high-error observations.
- Improve out-of-distribution detection.
- Build an interactive prediction interface.

## Dataset Credit

[Student Performance Factors Dataset — Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)


