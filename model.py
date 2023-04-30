import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# from pywaffle import Waffle
import matplotlib.pyplot as plt
sns.set()
import warnings
warnings.filterwarnings('ignore')
# load Datasets
df= pd.read_csv("C:/ML Model Deployment/dataset.csv")
print(df.head())
# dropping redundant columns having same info or irrelavant with job satisfaction
df= df.drop(['REF_DATE','DGUID','UOM','UOM_ID','VECTOR','COORDINATE','STATUS','SYMBOL','TERMINATED','SCALAR_ID','DECIMALS','SCALAR_FACTOR'],axis=1)
print(df.head(2))
# Dropping the records with VALUE missing in data dataframe.
df = df[~df.VALUE.isnull()].copy()

# checking the missing values
df.isnull().sum()
# new dataframe to remove repetion
df1=df.iloc[1::4]
print(df1.head(2))
df2=df1.copy()
df2.loc[df2['Response'] == "Don't know/refusal/not stated", 'Response'] ='unknown'
df2.head(10)
df2.rename(columns = {'Age group':'Age'}, inplace = True)
df2.rename(columns = {'Employment type':'Employment_type'}, inplace = True)
print(df2.head(2))
df5=df2.copy()
df5= df5.drop(['Estimates','VALUE'],axis=1)
print(df5.head(2))
df5.Response[(df5.Response == 'Very satisfied') | (df5.Response == 'Satisfied')|(df5.Response == 'Very satisfied or satisfied')|(df5.Response =='Total, satisfaction with work-home balance')] = 1
df5.Response[(df5.Response == 'Dissatisfied')|(df5.Response == 'Neither satisfied nor dissatisfied')|(df5.Response == 'Very dissatisfied')|(df5.Response == 'Dissatisfied or very dissatisfied')|(df5.Response == 'unknown')]= 0

print(df5.head(9))
inputs = df5[['GEO','Employment_type','Age','Sex']]
print(inputs.head(2))
target = df5['Response']
print(target.head(2))
target=target.astype('int')
print(target.head(2))
from sklearn.preprocessing import LabelEncoder
le_GEO = LabelEncoder()
le_Sex = LabelEncoder()
le_Employment_type = LabelEncoder()
le_Age = LabelEncoder()
inputs['GEO_n'] = le_GEO.fit_transform(inputs['GEO'])
inputs['Sex_n'] = le_GEO.fit_transform(inputs['Sex'])
inputs['Employment_type_n'] = le_GEO.fit_transform(inputs['Employment_type'])
inputs['Age_n'] = le_GEO.fit_transform(inputs['Age'])
print(inputs.head(2))
inputs_n = inputs.drop(['GEO','Sex','Age','Employment_type'],axis='columns')
print(inputs_n.tail(3))
from sklearn import tree
model = tree.DecisionTreeClassifier()
print(model.fit(inputs_n,target))
print(model.score(inputs_n,target))
print(model.predict([[2,0,1,8]]))
print(model.predict([[3,0,2,9]]))
import pickle
# Saving our model
file_name = 'model.pkl'

with open(file_name, 'wb') as file:
    pickle.dump(model,file)