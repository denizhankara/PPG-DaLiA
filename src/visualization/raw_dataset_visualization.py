#raw_dataset_visualization.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

path = "../data/raw/PPG_FieldStudy/S2/S2.pkl"


with open(path, "rb") as f:
    data = pickle.load(f, encoding="latin-1")

activity = pd.DataFrame(data["activity"]).astype(int)
activity.columns = ["Activity"]
dic_activity = {1: "Sitting", 2: "Stairs", 3: "Soccer", 4: "Cycling", 5: "Driving", 6: "Lunch", 7: "Walking", 8: "Working"}

label = pd.DataFrame(data["label"])
label = pd.DataFrame(np.repeat(label.values,8,axis=0))
label.columns = ["Label"]
if(np.size(label, axis = 0) < np.size(activity, axis = 0)):
    mean = label.mean()
    while(np.size(label, axis = 0) < np.size(activity, axis = 0)):
        label = label.append(mean, ignore_index=True)

signal = pd.DataFrame(data["signal"])
ACC = pd.DataFrame(signal["chest"].ACC)
ACC = ACC.iloc[::175, :]
ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
ACC.reset_index(drop = True, inplace=True)

ECG = pd.DataFrame(signal["chest"].ECG)
ECG = ECG.iloc[::175, :]
ECG.reset_index(drop = True, inplace=True)

Resp = pd.DataFrame(signal["chest"].Resp)
Resp = Resp.iloc[::175, :]
Resp.columns = ["Resp"]
Resp.reset_index(drop = True, inplace=True)

chest = pd.concat([ACC], sort=False)
chest["Resp"] = Resp
chest["ECG"] = ECG
chest.reset_index(drop=True, inplace=True)
chest = chest.add_prefix('chest_')

ACC = pd.DataFrame(signal["wrist"].ACC)
ACC = ACC.iloc[::8, :]
ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
ACC.reset_index(drop = True, inplace=True)

EDA = pd.DataFrame(signal["wrist"].EDA)
EDA.columns = ["EDA"]

BVP = pd.DataFrame(signal["wrist"].BVP)
BVP = BVP.iloc[::16, :]
BVP.columns = ["BVP"]
BVP.reset_index(drop = True, inplace=True)

TEMP = pd.DataFrame(signal["wrist"].TEMP)
TEMP.columns = ["TEMP"]

wrist = pd.concat([ACC], sort=False)
wrist["BVP"] = BVP
wrist["TEMP"] = TEMP
wrist.reset_index(drop = True, inplace=True)
wrist = wrist.add_prefix('wrist_')

signals = chest.join(wrist)

for k,v in data["questionnaire"].items() :
    signals[k] = v

signals = signals.join(activity)
signals = signals.join(label)

signals['Subject'] = data["subject"]




# HR vs PPG Data Plot 

fig, ax1 = plt.subplots(figsize=(10, 4))
style = dict(size=8, color='k')

i = 0
x_start = 0
x_end = 0
while(i < len(signals.loc[:, 'Activity'])):
    sport_index = signals.loc[i, 'Activity']
    
    if(sport_index != 0):
        
        x_start = i
        while(i < len(signals.loc[:, 'Activity'])):
            if(signals.loc[i, 'Activity'] != sport_index):
                break
            
            else:
                i += 1
            
        x_end = i-1
        sport = dic_activity[sport_index]
        plt.axvspan(xmin=x_start + 60, xmax=x_end, color='#e8e8e8')
        ax1.text((x_start+x_end)//2 - 250, 450, sport,rotation=45, **style)
        
        x_start = 0
        x_end = 0
    
    else :
        i += 1
print(signals.wrist_BVP.max(),signals.wrist_BVP.min())
ax1.set_ylim(top = signals.wrist_BVP.max(), bottom = signals.wrist_BVP.min())
signals.loc[:, 'wrist_BVP'].plot(ax=ax1,c='#b34a3c',label='PPG',linewidth=0.25)
plt.xlabel("Time", fontsize=12)
plt.ylabel("PPG Data", fontsize=12)

ax2 = ax1.twinx()
signals.loc[:, 'Label'].plot(ax=ax2,c='#3cb371',label='HR',linewidth=0.5)
plt.ylabel('Heart Rate',fontsize=12)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

plt.savefig('Figure_1_S2HRvsPPG.svg')


