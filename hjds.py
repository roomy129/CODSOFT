import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv("D:\\Titanic-Dataset.csv")


df["Age"] = df["Age"].fillna(df["Age"].median())
#print(df)
print("Tail datas")
print(df.tail(10))
print("Head datas")
print(df.head(10))

df["Age"] = df["Age"].astype(int)
#print(df.head(10))
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
print(df)

le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])       # male=1, female=0
df["Embarked"] = le.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
