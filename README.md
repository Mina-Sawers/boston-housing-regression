# 🏡 Boston Housing Price Predictor

An end-to-end Machine Learning web application that predicts real estate prices in Boston based on specific property features. 

🚀 **[Try the Live App Here](https://boston-housing-regression-mina-sawiris.streamlit.app/)**

## 📌 Overview
This project demonstrates the complete machine learning lifecycle: from data extraction and model tuning to web deployment. It features an interactive dashboard where users can adjust property features and instantly see the predicted home value alongside exploratory data insights.

## 🧠 The Machine Learning Model
The core of this application is a **Decision Tree Regressor**. 
During development, extensive hyperparameter tuning was conducted to handle the Bias-Variance tradeoff. The model was optimized with a `max_depth` of **4** to prevent **overfitting**, ensuring the algorithm understands the underlying market trends rather than simply memorizing the training data.

**Features Analyzed:**
*   **RM (Rooms):** Average number of rooms per dwelling.
*   **LSTAT (Poverty Rate):** Percentage of the lower status of the population.
*   **PTRATIO (Student-Teacher Ratio):** Pupil-teacher ratio by local town.

## 🛠️ Tech Stack
*   **Machine Learning:** Scikit-Learn, NumPy, Pandas, Joblib
*   **Web Framework:** Streamlit
*   **Data Visualization:** Plotly Express
*   **Deployment:** Streamlit Community Cloud

## 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/mina-sawiris/boston-housing-regression.git](https://github.com/mina-sawiris/boston-housing-regression.git)