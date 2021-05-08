# -*- coding: utf-8 -*-
import logging
from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import os


def processData(file, output_path):
    # file name
    current_subject = os.path.split(file)[-1]

    # read data
    signals = pd.read_csv(file)
    # subset x, y, z acceleration, measured heart rate and activity
    signals = signals[['window_ID','wrist_ACC_x','wrist_ACC_y','wrist_ACC_z','wrist_BVP', 'Activity']]
    # group signals by activity and window and select only those that have a full window (8)
    sig_activity = [x for _, x in signals.groupby(['Activity','window_ID']).filter(lambda x: len(x) == 8).groupby('Activity')]
    #print(sig_activity)

    # save processed data to appropriate path
    # filename w/o ext
    current_subject = os.path.splitext(current_subject)[0]
    # save grouped signals in individual files
    for x in sig_activity:
        dump_dir = os.path.join(output_path, current_subject + "_Activity_" + str(x['Activity'].iloc[0]) + ".csv")
        x.to_csv(dump_dir,index=False)

    # Give prompt
    print("Processing of " + current_subject + " is complete ! \n")

    pass


def cli_main():
    # Get all the patient data in raw folder
    data_path = "../../data/interim/PPG_FieldStudy_Windowed/"
    files = [f.path for f in os.scandir(data_path) if f.is_file() and 'labels' not in PurePath(f).name]
    
    
    #  Make output path for saving the processed results
    output_path = "../../data/interim/PPG_FieldStudy_Activities_Filtered/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #  Process each of the files and export to output path
    for file in files:
        #print(file)
        processData(file, output_path)

if __name__ == '__main__':
    cli_main()