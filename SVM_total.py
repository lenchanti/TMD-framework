import os
import csv
import shutil
import re
import numpy as np
from scipy import interpolate
import pandas as pd
#import tensorflow_decision_forests as tfdf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from glob import glob
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# mode_class={"manualChar":1,
#             "powerChar":2,
#             "walking":3,
#             "still":4,
#             "metro":5,
#             "car":6,
#             "bus":7}
lst=["D:\논문과제1\prep\\quarter_sec",
     "D:\논문과제1\prep\\half_sec",
     "D:\논문과제1\prep\\one_sec",
     "D:\논문과제1\prep\\two_sec",
     "D:\논문과제1\prep\\three_sec",
     "D:\논문과제1\prep\\four_sec"]
lst=[
     "D:\논문과제1\prep\\one_sec",
     "D:\논문과제1\prep\\half_sec",
    "D:\논문과제1\prep\\quarter_sec"
     ]


def z_normalize(data):
    return (data - np.mean(data)) / np.std(data)



# path="D:\논문과제1\prep\\quarter_sec"
#path="D:\논문과제1\prep\\half_sec"
#path="D:\논문과제1\prep\\one_sec"
#path="D:\논문과제1\prep\\two_sec"
#path="D:\논문과제1\prep\\three_sec"
#path="D:\논문과제1\prep\\four_sec"
# path="D:\논문과제1\prep\\five_sec"

for path in lst:

    
    df=pd.DataFrame()
    for files_path in os.listdir(path): 
        f=pd.read_csv(path+"\\"+files_path,low_memory=False)
        df=pd.concat([df,f],axis=0,ignore_index=True)



    file=df.iloc[:,1:]


    X=file.iloc[:,:-1]
    X = X.to_numpy()


    
    y=file.iloc[:,-1:]
    y = y.values.ravel() 

    #데이터 분할
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101,stratify=y)

    model = SVC(kernel='rbf', gamma=0.01, C=10, tol=0.001,decision_function_shape="ovo")
    
    model.fit(X_train, y_train)
    
    
    #예측
    y_pred = model.predict(X_test)
    
    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)

    p=path.replace("D:\논문과제1\prep\\","")
    print(f"{p}_Accuracy: {accuracy * 100:.2f}%")
    
