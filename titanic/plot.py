import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #Using seaborn :) 

df= pd.read_csv('kaggle_titanic/train.csv', sep = '\s*,\s*', na_values= ".",engine = 'python')
print (df)    #print data
(df['Name']) #test selector :)
print(df.columns.tolist()) #Print only columns
df.shape
df['Age'].fillna(value=-1,inplace=True)

sns.distplot(df['Age'])
plt.title('Age Distribution of all passengers')
plt.show()  

#Create Scatterplots for Various Combinations in the Dataset
#Age vs Survived
sns.lmplot(x='Age', y= 'PassengerID', data=df)
plt.title('Survived vs Age')
plt.figure()
plt.show()
