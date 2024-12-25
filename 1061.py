# Import Libraries
# pip install pandas numpy matplotlib seaborn scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load Dataset
# Download the dataset from Kaggle and place it in the same directory
data = pd.read_csv("creditcard.csv") #link of dataset Download https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 

# Data Exploration
print(data.head())
print(data.info())
print(data['Class'].value_counts())  # Check the distribution of classes

# Class Distribution Visualization
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Feature Scaling (Amount column)
from sklearn.preprocessing import StandardScaler
data['Amount'] = StandardScaler().fit_transform(data[['Amount']])

# Dropping unnecessary columns
data = data.drop(columns=['Time'], axis=1)  # Time is irrelevant for the model

# Balancing the Dataset (Undersampling)
fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0].sample(len(fraud))  # Undersample the majority class

balanced_data = pd.concat([fraud, non_fraud], axis=0)
print("Balanced Dataset Class Distribution:")
print(balanced_data['Class'].value_counts())

# Splitting the Data
X = balanced_data.drop(columns=['Class'])
y = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluating the Model
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Visualization of Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Example Prediction
example_transaction = X_test.iloc[0].values.reshape(1, -1)  # Example transaction
prediction = model.predict(example_transaction)
print("Predicted Class for the Example Transaction:", "Fraud" if prediction[0] == 1 else "No Fraud")