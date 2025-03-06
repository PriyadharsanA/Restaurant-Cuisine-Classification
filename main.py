import pandas as pd
import numpy as np
#Data importing and initial preprocessing
df=pd.read_csv("C:/Users/User/Desktop/Dataset.csv").dropna()
df=df.drop(columns=['Restaurant Name','Address','Country Code','Locality','Locality Verbose','Longitude','Latitude'])
df=df.drop_duplicates()

from sklearn.preprocessing import LabelEncoder
#Label Encoding
le=LabelEncoder()
df['Has Table booking']=le.fit_transform(df['Has Table booking'])    
df['Has Online delivery']=le.fit_transform(df['Has Online delivery'])
df['Is delivering now']=le.fit_transform(df['Is delivering now'])
df['Switch to order menu']=le.fit_transform(df['Switch to order menu'])
df['City']=le.fit_transform(df['City'])
df['Currency']=le.fit_transform(df['Currency'])
df['Rating color']=le.fit_transform(df['Rating color'])
df['Rating text']=le.fit_transform(df['Rating text'])

from sklearn.preprocessing import MultiLabelBinarizer
#Multi Label Binarizer for Cuisines
mlb=MultiLabelBinarizer()
df['Cuisines']=mlb.fit_transform(df['Cuisines'])

from sklearn.preprocessing import StandardScaler
#Scaling data values
sc=StandardScaler()
df['Average Cost for two']=sc.fit_transform(df[['Average Cost for two']])
df['Votes']=sc.fit_transform(df[['Votes']])

#Initializing X & Y
x=df.drop(columns='Cuisines')
y=df['Cuisines']

from sklearn.model_selection import train_test_split
#Split the data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Importing all models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

#Initializing models(Using ONE V/S REST AS IT IS A MULTI-CLASS PROBLEM)
lr=OneVsRestClassifier(LogisticRegression())
knn=OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
rf=OneVsRestClassifier(RandomForestClassifier(n_estimators=5))

#Model Training and Accuracy Score
lr.fit(x_train,y_train)
y_pred1=lr.predict(x_test)

knn.fit(x_train,y_train)
y_pred2=knn.predict(x_test)

rf.fit(x_train,y_train)
y_pred3=rf.predict(x_test)


#Performance Metrics
from sklearn.metrics import classification_report
print("1.LOGISTIC REGRESSION")
print("Accuracy Score: ",lr.score(x_test,y_test)*100)
print(classification_report(y_test,y_pred1,zero_division=1))
print('\n')
print("2.K-NEIGHBORS CLASSFIFER")
print("Accuracy Score: ",knn.score(x_test,y_test)*100)
print(classification_report(y_test,y_pred2,zero_division=1))
print('\n')
print("3.RANDOM FOREST")
print("Accuracy Score: ",rf.score(x_test,y_test)*100)
print(classification_report(y_test,y_pred3,zero_division=1))
