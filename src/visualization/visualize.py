# -*- coding: utf-8 -*-
import logging
from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import re



def plotData(file, output_path):
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


    # take short time Fourier transform of the relevant columns
    for var in ['wrist_ACC_x','wrist_ACC_y','wrist_ACC_z','wrist_BVP']:
        #print("Processing of " + current_subject + " " + var + "\n")
        # _ , _, Sxx = signal.stft(sig_window[0][var], fs=8, nfft=2048, nperseg=8)
        f, t, Sxx = signal.spectrogram(signals[var][12000:15000], fs=8, nfft=2048, nperseg=8)


        plt.pcolormesh(t, f, Sxx, shading='auto')

        plt.title(var)

        plt.ylabel('Frequency [Hz]')

        plt.xlabel('Time [sec]')

        plt.savefig(output_path + current_subject + var + '.png')
    # Give prompt
    print("Processing of " + current_subject + " is complete ! \n")

    pass


def cli_main():
    # Get all the patient data in raw folder
    data_path = "../../data/interim/PPG_FieldStudy_Windowed_Activity_Recognition/"
    # find all files in folder
    files = [f.path for f in os.scandir(data_path) if f.is_file() and 'labels' not in PurePath(f).name]
    # sort in order by subject (numerical part)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    #  Make output path for saving the processed results
    output_path = "../../reports/figures/spectra/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #  Process each of the files and save to output path
    for file in files:
        plotData(file, output_path)
    #plotData(files[0], output_path)
  
    

if __name__ == '__main__':
    cli_main()