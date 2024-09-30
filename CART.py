import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, precision_recall_fscore_support

lst=["D:\논문과제1\prep\\quarter_sec",
      "D:\논문과제1\prep\\half_sec",
     # "D:\논문과제1\prep\\one_sec",
     # "D:\논문과제1\prep\\two_sec",
     # "D:\논문과제1\prep\\three_sec",
     # "D:\논문과제1\prep\\four_sec",
     # "D:\논문과제1\prep\\five_sec",
     "D:\논문과제1\prep_3rd\quarter"
     ]

for path in lst:
    df=pd.DataFrame()
    for files_path in os.listdir(path): 
        f=pd.read_csv(path+"\\"+files_path,low_memory=False)
        df=pd.concat([df,f],axis=0,ignore_index=True)


    file=df.iloc[:,1:]

    X=file.iloc[:,:-1]
    y=file.iloc[:,-1:]
    mode_class=set(y["Mode"])
    y=pd.get_dummies(y,columns=["Mode"])
    
    
    #데이터 분할
    X_train,X_test=train_test_split(X,test_size=0.3,random_state=42)
    y_train,y_test=train_test_split(y,test_size=0.3,random_state=42)
    

    #결정 트리 모델 생성 및 학습 
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    
    #예측
    y_pred = clf.predict(X_test)
    
    # 정확도 출력
    accuracy = accuracy_score(y_test, y_pred)
    p=path.replace("D:\논문과제1\prep\\","")
    print(f"{p}_Accuracy: {accuracy * 100:.2f}%")
    
    
    report = classification_report(y_test, y_pred, target_names=mode_class)
    print(report)

    #precision_recall_fscore_support 사용
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')



