# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import Necessary Libraries and Load Data.

Step 3: Split Dataset into Training and Testing Sets.

Step 4: Train the Model Using Stochastic Gradient Descent (SGD).

Step 5: Make Predictions and Evaluate Accuracy.

Step 6: Generate Confusion Matrix.

STEP 7: Stop the program.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Pradeep E
RegisterNumber: 212223230149
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, Y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")

cf = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix")
print(cf)


```

## Output:
![Screenshot 2024-10-18 142202](https://github.com/user-attachments/assets/d5d8ccfd-cfb9-41dc-8b6f-8442d3656d3e)

![Screenshot 2024-10-18 142216](https://github.com/user-attachments/assets/f864e599-a1a3-4285-adf0-c8050fd75a42)

![Screenshot 2024-10-18 142220](https://github.com/user-attachments/assets/664efb74-cd71-4e02-b79e-48d7e07c2894)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
