import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Current working directory:", os.getcwd())

file_path = 'D:\codsoft\Titanic-Dataset.csv'

if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
else:
    titanic_df = pd.read_csv(file_path)

    titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
    titanic_df.drop(columns=['Cabin'], inplace=True)
    titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)
    titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
    titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'], drop_first=True)

    X = titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
    y = titanic_df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
