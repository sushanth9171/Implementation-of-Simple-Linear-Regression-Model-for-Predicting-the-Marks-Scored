# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Sushanth
RegisterNumber:  212225230088
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)
print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]]  
y = df["Marks_Scored"]     
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}"
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="531" height="708" alt="Screenshot 2026-01-30 120025" src="https://github.com/user-attachments/assets/b6d72b7c-8a0c-452b-a8a3-be0549926fdf" />
<img width="832" height="579" alt="Screenshot 2026-01-30 120109" src="https://github.com/user-attachments/assets/95024f1c-b2b0-46e0-ae3c-c392e18dc1fe" />
<img width="824" height="121" alt="Screenshot 2026-01-30 120145" src="https://github.com/user-attachments/assets/578ee7ce-7abe-4f03-812e-50f9d22a0d55" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
