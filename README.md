# Voting Classifier on the UCI Adult Dataset

This project applies **Random Forest**, **Support Vector Machine (SVM)**, and **XGBoost** classifiers using both hard and soft voting ensembles to classify the UCI Adult dataset. The dataset aims to predict whether a person earns over $50K per year based on various demographic attributes.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Project Setup](#project-setup)
3. [Code Walkthrough](#code-walkthrough)
4. [Results](#results)
5. [Dependencies](#dependencies)
6. [How to Run](#how-to-run)

---

## Dataset Description

The dataset used is the **UCI Adult dataset**, which contains information about individuals, such as:

- **Age**
- **Workclass**
- **Education**
- **Marital Status**
- **Occupation**
- **Relationship**
- **Race**
- **Sex**
- **Hours-per-week**
- **Native Country**

The target variable is **income**, indicating whether the individual's income is **">50K"** or "<=50K".

---

## Project Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Code Walkthrough

1. **Import Libraries**:
   - `pandas`, `numpy` for data manipulation.
   - `matplotlib`, `seaborn` for visualization.
   - `scikit-learn` for preprocessing and model implementation.
   - `xgboost` for the XGBoost classifier.

2. **Fetch Dataset**:
   ```python
   from ucimlrepo import fetch_ucirepo
   adult = fetch_ucirepo(id=2)
   X = adult.data.features
   y = adult.data.targets
   ```

3. **Preprocessing**:
   - Handle missing values by filling them with the mode.
   - Convert categorical variables to numerical using one-hot encoding.
   - Standardize numerical features using `StandardScaler`.

4. **Train-Test Split**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

5. **Voting Classifiers**:
   - **Random Forest**
   - **Support Vector Machine (SVM)**
   - **XGBoost**

   Both **hard voting** and **soft voting** classifiers are implemented:

   ```python
   from sklearn.ensemble import VotingClassifier
   
   hard_voting_clf = VotingClassifier(
       estimators=[('rf', rf_clf), ('svm', svm_clf), ('xgb', xgb_clf)],
       voting='hard'
   )
   
   soft_voting_clf = VotingClassifier(
       estimators=[('rf', rf_clf), ('svm', svm_clf), ('xgb', xgb_clf)],
       voting='soft'
   )
   ```

6. **Evaluation**:
   - Accuracy score
   - Classification report

---


## Dependencies

- Python 3.7+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `ucimlrepo`

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ucimlrepo
```

---

## How to Run

1. Ensure you have all the dependencies installed.
2. Run the script:

   ```bash
   python voting_classifier.py
   ```

3. View the accuracy and classification report in the console output.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

[Your Name](https://github.com/your-username)

---
