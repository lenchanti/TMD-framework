import os
import pandas as pd


def DeleteAllfiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return"REMOVE ALL FILE"
    else:
        return "Directory Not Found"
    
    
  
#데이터 읽어오기
path="D:\논문과제1\crc1st_data_1_60"
# path="D:\논문과제1\crc1st_data_60_119"
print(DeleteAllfiles(path+"\\merge"))
print(os.listdir(path))
files_list=[]
timeline=pd.DataFrame({"Time":[]})

#60HZ 주파수로 resampling
count=0
for j in range(30,630):
    for i in range(1,61):
        count+=1
        timeline.loc[count-1]=(j+round(i*0.0166,4))
        #print(count)
        
num=0
for folders in os.listdir(path):
    if "#" in folders:
        num=num+1

        if(num>115): #115 명의 참여자
            break

        _path=path
        _path=path+"\\"+folders

        os.chdir(_path)
        merge_df=pd.DataFrame()
        file_num=0
        file_count=0
        for file_path in os.listdir(_path):
            
            file_num=len((os.listdir(_path)))
            file_count+=1
            print(file_num)
            print(file_count)
            if 'SensorData' in file_path:



            #file_df란 raw data를 dataframe읋 불러 오는 것
            #file_concat_df란 linear interpolate를 하기 위해 
            #60HZ의 값을 가진 data frame을 덧 붙여 준것
            
            #linear interpolation으로 resampling
                
                file_df=pd.read_csv(_path+"\\"+file_path,low_memory=False)
                idf=file_df.iloc[:,-2:]
                
                idf=pd.concat([idf["Mode"].fillna(idf["Mode"][0]),
                          idf["Survay1"].fillna(idf["Survay1"][0])]
                              ,axis=1)
                idf=idf.iloc[:36000]
            
                file_concat_df=pd.concat([file_df.iloc[:,:-2],timeline],ignore_index=True)
                file_concat_df=file_concat_df.sort_values(by="Time")
                
                file_concat_df=file_concat_df.interpolate()
                
                file_concat_df=file_concat_df.reset_index()
                file_concat_df=file_concat_df.sort_values(by="index")
                file_concat_df=file_concat_df.set_index(["index"])
                
                df=file_concat_df[len(file_df):len(file_concat_df)]
                df=df.interpolate()
                df=df.reset_index(drop=True)
                
                df = df[['GraX', 'GraY', 'GraZ',
                     'LAccX', 'LAccY', 'LAccZ',
                     'GyroX', 'GyroY', 'GyroZ',
                     'MagX', 'MagY', 'MagZ'
                     ]]
                df=pd.concat([df,idf],axis=1)
                
            #여기서 df 란 선형 보간을 완료한 우리가 원하는 값의  dataframe
            
            
            #data preprocessing을 위해 앞 뒤 30s씩 날린다
                # df=df.iloc[1800:len(df)-1800,]
                
                df=df.reset_index(drop=True)
               
                merge_df=pd.concat([merge_df,df],axis=0,ignore_index=True)
                
            if (file_count==(file_num)):
                os.chdir(path+"\\merge")
                merge_df.to_csv(folders.replace("#", "")+"_merge.csv",index=False)
                    
                    
                

    
    
    
    
    
    
    

