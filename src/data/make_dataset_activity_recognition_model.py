# -*- coding: utf-8 -*-
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

def plot_confusion_matrix_2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    y = np.repeat(np.arange(0,len(classes)),15)
    plt.xlim(-0.5, len(np.unique(y))-0.5)
    plt.ylim(len(np.unique(y))-0.5, -0.5)
    plt.tight_layout()


def findActivityAndEncode(signals,current_subject):
    # encode the necessary fields with CategoryBooster's encoder, which prevents data leaks on windows

    features_for_activity_recognition = ['wrist_ACC_x',
 'wrist_ACC_y',
 'wrist_ACC_z',
 'wrist_BVP',
 'wrist_TEMP']

    X = signals[features_for_activity_recognition].values
    y = signals.Activity

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train);
    y_pred = rf.predict(X_test)

    score = rf.score(X_test, y_test)
    print(score)

    from sklearn.metrics import plot_confusion_matrix

    plot_confusion_matrix(rf, X_test, y_test,normalize='true')
    plt.title('Activity Score: '+ str(score))

    output_path = "../../reports/figures/activity_recognition/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(output_path + current_subject)

    # predict the activity
    predicted_activities = rf.predict(X)
    signals['predicted_activity'] = predicted_activities

    # remove the unnecessary signals
    del signals["WEIGHT"]
    del signals["Gender"]
    del signals["AGE"]
    del signals["HEIGHT"]
    del signals["SKIN"]
    del signals["SPORT"]
    del signals["Activity"]

    cols = ["predicted_activity"]

    # Define train and target
    target = signals[['Label']]
    signals = signals.drop('Label', axis=1)

    # Define catboost encoder
    cbe_encoder = CatBoostEncoder(cols=cols)

    # Fit encoder and transform the features
    cbe_encoder.fit(signals, target)
    signals = cbe_encoder.transform(signals)

    signals['Label'] = target
    gc.collect()
    return signals


def rollWindows(signals):
    """
    Make distinct windows of 8 seconds, which are sliding

    Since target labels are made by taking "mean" of the 8 second heartbeat windows,
    the current sliding windows will provide more data and smoother transition to window changes

    """

    # First, add time as seconds in 4 Hz as ordering column

    signals.reset_index(level=0, inplace=True)

    #signals=signals.iloc[:1000]

    signals = roll_time_series(signals, column_id="Subject", column_sort="index", max_timeshift=7, min_timeshift=7)

    signals['window_ID'] = signals['id'].apply(lambda x1: x1[0] + "_" + str(x1[1]))

    # Put the window ID to first place and remove excess id column
    del signals['id']
    del signals['Subject']  #  we have subject embedded in window_ID now

    #  Reorder to get the window ID to first column
    cols = list(signals.columns)
    cols = [cols[-1]] + cols[:-1]
    signals = signals[cols]

    # make the window labels by taking the mean of HR over the 8 second sliding window
    window_labels = signals.groupby(['window_ID']).mean()['Label']
    return signals,window_labels


def processData(subfolder, output_path):
    # patient_path = "../../data/raw/PPG_FieldStudy/S1/S1.pkl"
    current_subject = os.path.split(subfolder)[-1]

    patient_path = os.path.join(subfolder, current_subject + ".pkl")

    signals = load_data(patient_path)
    #signals =signals[:10000]
    signals = findActivityAndEncode(signals,current_subject)
    signals,window_labels = rollWindows(signals)

    # save processed data to appropriate path

    dump_dir = os.path.join(output_path, current_subject + ".csv")
    signals.to_csv(dump_dir,index=False)

    # Save the labels of the windows to the appropriate path

    # add the predicted activities to label file as well for brevity
    del signals['Label']

    window_labels = pd.DataFrame(window_labels)
    window_labels = pd.merge(window_labels, signals, left_on=window_labels.index, right_on='window_ID')
    window_labels = window_labels[['window_ID', 'Label', 'predicted_activity']]
    window_labels.drop_duplicates(subset=['window_ID'], keep='first', inplace=True)


    window_labels_dump_dir = os.path.join(output_path,current_subject+ "_labels.csv")
    window_labels.to_csv(window_labels_dump_dir)

    # Give prompt
    print("Processing of " + current_subject + " is complete ! \n")

    pass


def cli_main():
    # Get all the patient data in raw folder
    data_path = "../../data/raw/PPG_FieldStudy/"
    subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]

    #  Make output path for saving the processed results
    output_path = "../../data/interim/PPG_FieldStudy_Windowed_Activity_Recognition/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #  Process each of the files and export to output path
    for subfolder in subfolders:
        processData(subfolder, output_path)


if __name__ == '__main__':
    cli_main()
