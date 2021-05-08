# -*- coding: utf-8 -*-
import logging
from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import os
from scipy import signal
import pickle
from tqdm import tqdm
from scipy.stats import zscore
import torch



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def processData(file, output_path):
    """
    input: file - file to be processed
            output_path - file to save processed data
    """

    # current file to process
    current_subject = os.path.split(file)[-1]
    fname=os.path.splitext(current_subject)[0]
    #print(fname)

    # read data from csv file
    signals = pd.read_csv(file)
    
    #print(signals.shape)

    # group signals by activity
    sig_window = [x for _, x in signals.groupby('window_ID')]
   
    # dictionary of lists to save window transforms
    dictlist = {}

    # loop over all window_IDs
    for x in tqdm(sig_window, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        # list of dataframes to store results
        listSxx = []
        # take short time Fourier transform of the relevant columns
        for var in ['wrist_ACC_x','wrist_ACC_y','wrist_ACC_z','wrist_BVP']:
            #print("Processing of " + current_subject + " " + var + "\n")
            _ , _, Sxx = signal.stft(sig_window[0][var], fs=8, nfft=2048, nperseg=8)
            # calculate magnitude, transpose, and convert to DataFrame
            Sxx = pd.DataFrame(np.transpose(np.abs(Sxx)))
            # z-score normalization by row
            Sxx = Sxx.apply(zscore, axis=1)
            # convert transformed data to list of tensors
            list_of_tensors = [torch.tensor(df, dtype=torch.float32) for df in Sxx]
            # stack tensors to get tyhe right sshape (3, 1025)
            tstack = torch.stack(list_of_tensors)
            # append dataframes to the list
            listSxx.append(tstack)
        # stack the channel tensors - yields a tensor (4, 3, 1025)
        cstack = torch.stack(listSxx)
        # add stacked channels to the dictionary with window_ID as key
        dictlist[str(x['window_ID'].iloc[0])] = cstack
    
    # save processed data to appropriate path  
    dump_file = os.path.join(output_path, fname+".pkl")
    save_object(dictlist,dump_file)

    # Give prompt
    print("Processing of " + current_subject + " is complete ! \n")

    pass


def cli_main():
    # Get all the patient data in raw folder
    data_path = "../../data/interim/PPG_FieldStudy_Windowed_Activity_Recognition/"
    # find all files in folder
    files = [f.path for f in os.scandir(data_path) if f.is_file() and 'labels' not in PurePath(f).name]
    # sort them by time
    files.sort(key=lambda x: os.path.getmtime(x))
    
    
    #  Make output path for saving the processed results
    output_path = "../../data/interim/PPG_FieldStudy_CNN_Input/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #  Process each of the files and save to output path
    for file in files:
        processData(file, output_path)
    #processData(files[0], output_path)
  
    

if __name__ == '__main__':
    cli_main()
