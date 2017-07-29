import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df= pd.read_csv('kaggle_titanic/train.csv', sep = '\s*,\s*', na_values= ".",engine = 'python')
print (df)    #print data
(df['Name']) #test selector :)
print(df.columns.tolist()) #Print only columns
df.shape

#Visualize Data Using Scatterplot
#Plot 1: Survival,Pclass
#Plot 2: Survival, Age
#Plot 3: Survival, Fare Paid
groups = df.groupby('Survived')
fig,ax =plt.subplots()
for name,group in groups:
    ax.plot(group.Age,group.Survived,marker = 'o',linestyle=' ',ms=12,label = name)
ax.legend()
plt.show()

#TODO: Fix Indexing
#TODO: Add linear regression model
