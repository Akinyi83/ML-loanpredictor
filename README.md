# 🏦 Loan Approval Predictor

##  Overview
This project involves building a machine learning model that predicts whether a loan application should be **approved** or **rejected** based on key applicant details. It walks through the complete machine learning lifecycle — from data loading and cleaning to feature selection, model training and evaluation.

>  Dataset: [Loan Approval Dataset - Kaggle](https://www.kaggle.com/datasets/granjithkumar/loan-approval-data-set)

---

##  Project Goals
- Predict loan approval using a **binary classification model**
- Handle missing data, encode categorical variables, and scale numeric ones
- Perform **feature selection** using Random Forest
- Train and evaluate two models:
  - Logistic Regression
  - Decision Tree Classifier
- Compare model performance using accuracy, precision, recall, and F1-score

---

##  Tech Stack
- Python 3.x
- Jupyter Notebook / VS Code
- Libraries:
  - `pandas`, `numpy`, `seaborn`, `matplotlib`
  - `scikit-learn` (LogisticRegression, DecisionTreeClassifier, RandomForest, SelectFromModel)

---

## 📁 Project Structure
LoanApprovalProject/
│
├── loan-predictor.ipynb # Main notebook with full pipeline
├── Loan_train.csv # Dataset (Kaggle)
├── requirements.txt # Python dependencies
├── README.md # This file
└── Report.pdf # Final report


---

##  Key Steps

###  1. Data Preprocessing
- Handled missing values in `Gender`, `Married`, `LoanAmount`, etc.
- Cleaned `Dependents` column (e.g., “3+” → 3)
- Label encoded categorical features
- Standardized numeric features

###  2. Feature Selection
- Used `RandomForestClassifier` for feature importance ranking
- Selected top features using `SelectFromModel`

###  3. Model Training
- Trained Logistic Regression and Decision Tree classifiers
- Used 80/20 train-test split for evaluation

###  4. Evaluation

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 78.9%    | 76%       | **98.8%** | **0.86**   |
| Decision Tree       | 69.9%    | 76%       | 77.5%  | 0.77     |

📌 **Logistic Regression** performed best, with very high recall and F1-score.

---

##  Final Recommendation
We recommend **Logistic Regression** for this problem due to:
- High **recall** (minimizing rejection of eligible applicants)
- Better generalization than Decision Tree
- Strong **F1-score** and balanced performance across all metrics

---

##  Learnings & Takeaways
- Handling imbalanced datasets and noisy features is key to reliable predictions
- Feature selection can improve model accuracy and training efficiency
- Logistic Regression remains a powerful and interpretable baseline model

---

##  Authors
- KCBF Team — Group Assignment: Vallary Ogolla, Emmanuel Kimeu, Fredrick Njoroge, Nancy Kamau, Perpetual Muthaka
- Built for educational purposes under the Adanian Labs Data Science and AI Lab Initiative

---


