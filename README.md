# Movie Rating Prediction with Python

## 📌 Overview
This project predicts movie ratings based on features such as genre, director, and main actors. Using machine learning regression techniques, the model analyzes historical data to estimate user and critic ratings.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib

## 🚀 Workflow & Methodology
1. **Data Preprocessing:** Dropped missing target values (`Rating`) and imputed missing text features with `'Unknown'`.
2. **Feature Encoding:** Applied `TargetEncoder` to convert high-cardinality text data (Directors, Actors, Genres) into numeric representations based on target mean values.
3. **Data Splitting:** Divided data into an 80% training set and a 20% testing set using `train_test_split`.
4. **Model Training:** Trained a `RandomForestRegressor` model to handle non-linear relationships.
5. **Model Evaluation:** Measured performance using Root Mean Squared Error (RMSE) and R² Score.

## 📊 Results
* **Root Mean Squared Error (RMSE):** [Insert your RMSE score here, e.g., 0.82]
* **R² Score:** [Insert your R² score here, e.g., 0.45]

## 📁 Project Structure
