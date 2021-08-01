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
from tsfresh import extract_features
from functools import reduce,partial
import gc

considered_features = {"cwt_coefficients": [{"widths": width, "coeff": coeff, "w": w} for
                                 width in [(2, 5, 10, 20)] for coeff in range(15) for w in (2, 5, 10, 20)],
            }
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

    # sig_window=sig_window[:10]

    """
    # dictionary of lists to save window transforms
    dictlist = {}

    # sig_window=sig_window[:10]


    # loop over all window_IDs
    for x in tqdm(sig_window, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        # list of dataframes to store results
        listSxx = []
        # take short time Fourier transform of the relevant columns
        for var in ['wrist_ACC_x','wrist_ACC_y','wrist_ACC_z','wrist_BVP']:
            #print("Processing of " + current_subject + " " + var + "\n")
            _ , _, Sxx = signal.stft(x[var], fs=8, nfft=2048, nperseg=8)
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
        pass
    
    
    # save processed data to appropriate path  
    dump_file = os.path.join(output_path, fname+".pkl")
    save_object(dictlist,dump_file)
    """

    # declare diclist
    dictlist = {}

    # Loop again to extract and prepare added features

    sig_window= pd.concat(sig_window)

    listSxx = []

    for var in ['wrist_ACC_x', 'wrist_ACC_y', 'wrist_ACC_z', 'wrist_BVP']:
        columns = ['window_ID','index'] + [var]

        df = sig_window[columns]
        settings = {"cwt_coefficients": considered_features["cwt_coefficients"]}


        # without stacked dataframe
        extracted_features = extract_features(df, column_id='window_ID', column_sort='index',
                                              default_fc_parameters=settings,n_jobs=1)  # n_jobs=0

        extracted_features.fillna(0,inplace=True)

        #end = time.time()
        #print("Extracting: " + feature_file_name + " took" + str(end - start) + " seconds")

        # (4,1,60) stacked channel tensor
        #_, _, Sxx = signal.stft(sig_window[0][var], fs=8, nfft=2048, nperseg=8)
        # calculate magnitude, transpose, and convert to DataFrame
        #Sxx = pd.DataFrame(np.transpose(np.abs(Sxx)))
        # z-score normalization by row
        #Sxx = Sxx.apply(zscore, axis=1)
        # convert transformed data to list of tensors
        #list_of_tensors = [torch.tensor(df, dtype=torch.float32) for df in Sxx]
        list_of_tensors = torch.tensor(extracted_features.values, dtype=torch.float32)
        # stack tensors to get tyhe right sshape (3, 1025)
        # tstack = torch.stack(list_of_tensors)
        # append dataframes to the list
        listSxx.append(list_of_tensors)
        gc.collect()
        pass

    print("Start stacking")
    # stack the channel tensors - yields a tensor (4, 3, 1025)
    cstack = torch.stack(listSxx)

    print("Start adding to dict")
    # add stacked channels to the dictionary with window_ID as key
    for i,window_ID in enumerate(list(extracted_features.index)):
        dictlist[window_ID] = cstack[:,i,:].unsqueeze(1)

    pass
    print("Start dumping")
    # do cwt extraction on second loop,save a seperate pickle and load in dataloader
    # save processed data to appropriate path
    dump_file = os.path.join(output_path, fname + "_cwt.pkl")
    #save_object(dictlist, dump_file)
    torch.save(dictlist, dump_file)
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
