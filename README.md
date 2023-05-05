# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
  1.mport the standard libraries.

  2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

  3.Import LabelEncoder and encode the dataset.
  
  4.Import LogisticRegression from sklearn and apply the model on the dataset.
  
  5.Predict the values of array.
  
  6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
  
  7.Apply new unknown values 


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: praveen s
RegisterNumber:  212222240077

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])








*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![image](https://user-images.githubusercontent.com/119091638/233590073-fc8a1fd0-bb40-4a6b-b0ae-12538fb67588.png)



![image](https://user-images.githubusercontent.com/119091638/233590318-491a7fe9-0c73-45a3-939b-402a6bd879a8.png)


![image](https://user-images.githubusercontent.com/119091638/233590517-3ab6220d-a10a-4654-94be-3186e6edcd58.png)



![image](https://user-images.githubusercontent.com/119091638/233590447-f4af431a-7ebd-40d9-b1e3-542a5df53b88.png)


![image](https://user-images.githubusercontent.com/119091638/233590660-e6f4dae9-c4b0-42fa-bee5-6879ee349b3b.png)


![image](https://user-images.githubusercontent.com/119091638/233590929-7c9d8f5d-f8b4-4d50-9b4c-2621fcab3e77.png)
![image](https://user-images.githubusercontent.com/119091638/233591057-c311d8ec-7bf0-4d0a-b8d9-21892152ca13.png)
![image](https://user-images.githubusercontent.com/119091638/233591160-4d5d1063-194d-4f76-8f34-6738fddee8dd.png)
![image](https://user-images.githubusercontent.com/119091638/233591656-b847fb69-fb47-4a71-b486-bb0d5598e14d.png)
![image](https://user-images.githubusercontent.com/119091638/233591763-10b7cd6b-2c16-48ae-8ac9-8494c6a4b50f.png)
![image](https://user-images.githubusercontent.com/119091638/233591872-f786bc55-3d55-4c0d-a7f6-d6f0bba3b84a.png)
![image](https://user-images.githubusercontent.com/119091638/233591936-929d46c7-1c62-40f7-ac91-75febc0a2ab5.png)










## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
