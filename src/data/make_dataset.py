# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from category_encoders.cat_boost import CatBoostEncoder


import tsfresh
from tsfresh.utilities.dataframe_functions import roll_time_series


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin-1")

    signal = pd.DataFrame(data["signal"])
    ACC = pd.DataFrame(signal["chest"].ACC)
    ACC = ACC.iloc[::175, :]
    ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
    ACC.reset_index(drop=True, inplace=True)

    ECG = pd.DataFrame(signal["chest"].ECG)
    ECG = ECG.iloc[::175, :]
    ECG.reset_index(drop=True, inplace=True)

    Resp = pd.DataFrame(signal["chest"].Resp)
    Resp = Resp.iloc[::175, :]
    Resp.columns = ["Resp"]
    Resp.reset_index(drop=True, inplace=True)

    chest = pd.concat([ACC], sort=False)
    chest["Resp"] = Resp
    chest["ECG"] = ECG
    chest.reset_index(drop=True, inplace=True)
    chest = chest.add_prefix('chest_')

    ACC = pd.DataFrame(signal["wrist"].ACC)
    ACC = ACC.iloc[::8, :]
    ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
    ACC.reset_index(drop=True, inplace=True)

    EDA = pd.DataFrame(signal["wrist"].EDA)
    EDA.columns = ["EDA"]

    BVP = pd.DataFrame(signal["wrist"].BVP)
    BVP = BVP.iloc[::16, :]
    BVP.columns = ["BVP"]
    BVP.reset_index(drop=True, inplace=True)

    TEMP = pd.DataFrame(signal["wrist"].TEMP)
    TEMP.columns = ["TEMP"]

    wrist = pd.concat([ACC], sort=False)
    wrist["BVP"] = BVP
    wrist["TEMP"] = TEMP
    wrist.reset_index(drop=True, inplace=True)
    wrist = wrist.add_prefix('wrist_')

    signals = chest.join(wrist)
    for k, v in data["questionnaire"].items():
        signals[k] = v

    rpeaks = data['rpeaks']
    counted_rpeaks = []
    index = 0  # index of rpeak element
    time = 175  # time portion
    count = 0  # number of rpeaks

    while (index < len(rpeaks)):
        rpeak = rpeaks[index]

        if (rpeak > time):  # Rpeak appears after the time portion
            counted_rpeaks.append(count)
            count = 0
            time += 175

        else:
            count += 1
            index += 1
    # The rpeaks will probably end before the time portion so we need to fill the last portions with 0
    if (len(counted_rpeaks) < np.size(signals, axis=0)):
        while (len(counted_rpeaks) < np.size(signals, axis=0)):
            counted_rpeaks.append(0)
    peaks = pd.DataFrame(counted_rpeaks)
    peaks.columns = ["Rpeaks"]
    signals = signals.join(peaks)

    activity = pd.DataFrame(data["activity"]).astype(int)
    activity.columns = ["Activity"]
    signals = signals.join(activity)

    label = pd.DataFrame(data["label"])

    label = pd.DataFrame(np.repeat(label.values, 8, axis=0))
    label.columns = ["Label"]
    if (np.size(label, axis=0) < np.size(activity, axis=0)):
        mean = label.mean()
        while (np.size(label, axis=0) < np.size(activity, axis=0)):
            label = label.append(mean, ignore_index=True)

    signals = signals.join(label)

    signals['Subject'] = data["subject"]

    gc.collect()

    return signals


def processChestData(data):


    signal = pd.DataFrame(data["signal"])
    ACC = pd.DataFrame(signal["chest"].ACC)

    ACC = ACC.iloc[::175, :]
    ACC.columns = ["ACC_x", "ACC_y", "ACC_z"]
    ACC.reset_index(drop=True, inplace=True)

    ECG = pd.DataFrame(signal["chest"].ECG)
    ECG = ECG.iloc[::175, :]
    ECG.reset_index(drop=True, inplace=True)

    Resp = pd.DataFrame(signal["chest"].Resp)
    Resp = Resp.iloc[::175, :]
    Resp.columns = ["Resp"]
    Resp.reset_index(drop=True, inplace=True)

    chest = pd.concat([ACC], sort=False)
    chest["Resp"] = Resp
    chest["ECG"] = ECG
    chest.reset_index(drop=True, inplace=True)
    chest = chest.add_prefix('chest_')

    pass



def encodeFields(signals):
    # encode the necessary fields with CategoryBooster's encoder, which prevents data leaks on windows

    cols = ["Gender","SKIN","SPORT","Activity"]

    # Define train and target
    target = signals[['Label']]
    signals = signals.drop('Lgabel', axis=1)

    # Define catboost encoder
    cbe_encoder = CatBoostEncoder(cols=cols)

    # Fit encoder and transform the features
    cbe_encoder.fit(signals, target)
    signals = cbe_encoder.transform(signals)


    signals['Label'] = target
    gc.collect()
    return signals

def rollWindows(signals):



    pass


if __name__ == '__main__':

    patient_path = "../../data/raw/PPG_FieldStudy/S1/S1.pkl"

    signals = load_data(patient_path)
    signals = encodeFields(signals)
    signals = rollWindows(signals)

    pass


    pass
    #processPatient(patient_path)
