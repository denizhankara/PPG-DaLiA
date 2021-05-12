#interim_dataset_visualization.py
import csv
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gc
from category_encoders.cat_boost import CatBoostEncoder
from glob import glob
import os
import seaborn as sns
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tsfresh
from tsfresh.utilities.dataframe_functions import roll_time_series


# Open file as dataframe
#file_path = '../data/interim/PPG_FieldStudy_Windowed/s1.csv'
file_path = '../data/interim/all.csv'
with open(file_path,'r') as f:
    data = pd.read_csv(f)
    print('File 1 Read')
'''
for i in range(2,16):
    file_path = '../data/interim/PPG_FieldStudy_Windowed/s' + str(i) + '.csv'
    with open(file_path,'r') as f:
        data2 = pd.read_csv(f)
        data.append(data2,ignore_index=True)
        print('File ', i, ' Read')


data.to_csv(path_or_buf='../data/interim/all.csv',index=False)
# Display and drop irrelevant columns 

file_path = '../data/interim/all.csv'
with open(file_path,'r') as f:
    data = pd.read_csv(f)
'''
data = data.drop(columns=['window_ID','index','Gender','AGE','HEIGHT','SKIN','SPORT','Activity','WEIGHT','Rpeaks'])
print(data.columns)
data.columns = ['Chest X', 'Chest Y', 'Chest Z', 'Chest Resp', 'Chest ECG', 'Wrist X','Wrist Y','Wrist Z','Wrist BVP/PPG','Wrist Temp','Heart Rate' ]
print(data)

X = data[['Chest X', 'Chest Y', 'Chest Z', 'Chest Resp', 'Chest ECG', 'Wrist X','Wrist Y','Wrist Z','Wrist BVP/PPG','Wrist Temp']]
Y = data["Heart Rate"]  


seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = XGBRegressor()
model.fit(X_train,y_train)

print(model)
ax = plot_importance(model)
ax.grid(False)
plt.show()

'''
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy_score = accuracy_score(y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''


'''
corr = data.corr()

mask = np.triu(np.ones_like(corr,dtype=bool))

data.columns = ['Subject','Activity']
print(data)



ax = sns.heatmap(corr,mask=mask,vmin=-1, vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200),square=True)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
plt.show()'''