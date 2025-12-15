# 🚀 Startup Success Prediction using Machine Learning

An **end-to-end Machine Learning project** that predicts whether a startup is **likely to succeed or fail** based on key business and funding indicators. This project is designed as a **startup-style decision support tool** for founders, investors, and accelerators.

---

## 📌 Project Overview

Startups face high uncertainty, and early decisions around funding, team size, and growth strategy can determine long-term success. This project leverages **machine learning classification models**, optimized using **GridSearchCV**, to analyze startup characteristics and predict outcomes.

The final model is deployed using **Streamlit**, providing a clean and intuitive web interface for non-technical users.

---

## 🎯 Aim

To build a machine learning-based system that predicts **startup success or failure** using historical startup data and optimized classification models.

---

## 🧠 Problem Statement

Many startups fail due to poor planning, insufficient funding, or weak execution. Investors and founders need a **data-driven approach** to assess startup viability. This project aims to classify startups as **successful or failed** based on measurable attributes.

---

## 📊 Dataset

* **Source:** Kaggle – Startup Success Prediction Dataset
* **Target Variable:** `status`

  * `1` → Acquired (Successful)
  * `0` → Closed (Failed)

### Key Features Used:

* Total funding raised (USD)
* Number of funding rounds
* Team size
* Business milestones achieved
* Average participants / users
* Business relationships / partnerships

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * pandas, numpy
  * scikit-learn
  * matplotlib, seaborn
  * joblib
* **Model:** Random Forest Classifier
* **Hyperparameter Tuning:** GridSearchCV
* **Deployment:** Streamlit

---

## 🔍 Machine Learning Pipeline

1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Feature selection and preprocessing
4. Train-test split
5. Feature scaling using StandardScaler
6. Model training using Random Forest
7. Hyperparameter tuning using GridSearchCV
8. Model evaluation
9. Model saving and deployment

---

## 📈 Model Optimization (GridSearchCV)

GridSearchCV is used to tune important hyperparameters such as:

* Number of trees (`n_estimators`)
* Maximum tree depth (`max_depth`)
* Minimum samples required to split a node

This improves model accuracy and generalization.

---

## 📊 Evaluation Metrics

* Accuracy Score
* Precision, Recall, F1-score
* Confusion Matrix
* Feature Importance Analysis

---

## 🖥️ Web Application (Streamlit)

The Streamlit app allows users to:

* Enter startup details
* Instantly predict startup success or failure
* View confidence probability

### Target Users:

* 🚀 Startup Founders
* 💼 Investors
* 🧠 Accelerators

---

## 📂 Project Structure

```
startup-success-prediction-ml/
│
├── app.py
├── requirements.txt
├── startup_success_model.pkl
├── scaler.pkl
├── startup data.csv
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/startup-success-prediction-ml.git
cd startup-success-prediction-ml
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Open the displayed local URL in your browser.

---

## 🏆 Results & Insights

* Startups with higher funding and clear milestones show higher success probability
* Team size and partnerships significantly influence outcomes
* The model provides explainable insights using feature importance

---

## 🌟 Future Enhancements

* Add industry and location-based predictions
* Integrate success probability visualization
* Deploy on Streamlit Cloud
* Extend model using XGBoost

---

## 📌 Conclusion

This project demonstrates how machine learning can be applied to **real-world startup and investment decision-making**. By combining classification models, GridSearchCV optimization, and an intuitive web interface, the system acts as a **practical decision support tool**.

---

## 💖 Author

Developed with a strong interest in **startups, entrepreneurship, and AI-driven decision making**.

---

⭐ If you find this project useful, feel free to star the repository!
