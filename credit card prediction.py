import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load the dataset
data = pd.read_csv("D:\\creditcard.csv")
print(data.head(10))

#checks dataset
data.info()
data.isnull().sum()

#checks fraud
print(data['Class'].value_counts())

#to splitting datas
X = data.drop('Class', axis=1)
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train model
model = LogisticRegression()
model.fit(x_train, y_train)

#prediction 
y_pred = model.predict(x_test)

#for evalution
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))