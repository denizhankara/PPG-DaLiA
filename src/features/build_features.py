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

from glob import glob
from itertools import product
import os
import multiprocessing.pool
from functools import reduce,partial
import time

from multiprocessing import Pool

from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features

available_features= {

            "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
            "c3": [{"lag": lag} for lag in range(1, 4)],
            "cid_ce": [{"normalize": True}, {"normalize": False}],
            "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
            "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
            "quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
            "autocorrelation": [{"lag": lag} for lag in range(10)],
            "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
            "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
            "number_cwt_peaks": [{"n": n} for n in [1, 5]],
            "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
            "binned_entropy": [{"max_bins": max_bins} for max_bins in [10]],
            "index_mass_quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
            "cwt_coefficients": [{"widths": width, "coeff": coeff, "w": w} for
                                 width in [(2, 5, 10, 20)] for coeff in range(15) for w in (2, 5, 10, 20)],
            "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
            "ar_coefficient": [{"coeff": coeff, "k": k} for coeff in range(10 + 1) for k in [10]],
            "change_quantiles": [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
                                 for ql in [0., .2, .4, .6, .8] for qh in [.2, .4, .6, .8, 1.]
                                 for b in [False, True] for f in ["mean", "var"] if ql < qh],
            "fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                                product(["real", "imag", "abs", "angle"], range(100))],
            "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
            "value_count": [{"value": value} for value in [0, 1, -1]],
            "range_count": [{"min": -1, "max": 1}, {"min": 1e12, "max": 0}, {"min": 0, "max": 1e12}],
            "approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]],
            "friedrich_coefficients": (lambda m: [{"coeff": coeff, "m": m, "r": 30} for coeff in range(m + 1)])(3),
            "max_langevin_fixed_point": [{"m": 3, "r": 30}],
            "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                             {"attr": "slope"}, {"attr": "stderr"}],
            "agg_linear_trend": [{"attr": attr, "chunk_len": i, "f_agg": f}
                                 for attr in ["rvalue", "intercept", "slope", "stderr"]
                                 for i in [5, 10, 50]
                                 for f in ["max", "min", "mean", "var"]],
            "augmented_dickey_fuller": [{"attr": "teststat"}, {"attr": "pvalue"}, {"attr": "usedlag"}],
            "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
            "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
            "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
            "linear_trend_timewise": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                                      {"attr": "slope"}, {"attr": "stderr"}],
            "count_above": [{"t": 0}],
            "count_below": [{"t": 0}],
            "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
            "fourier_entropy":  [{"bins": x} for x in [2, 3, 5, 10, 100]],
            "permutation_entropy":  [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]],
            "query_similarity_count": [{"query": None, "threshold": 0.0}],
            "matrix_profile": [{"threshold": 0.98, "feature": f} for f in ["min", "max", "mean", "median", "25", "75"]]}




current_considered_features = {"maximum": None,
                      "minimum": None,
                      "mean_abs_change": None,
                      "variation_coefficient": None,
                      "fft_coefficient": [{"coeff": k, "attr": a} for a, k in
                                          product(["real", "imag", "abs", "angle"], range(25))],

                      "sum_of_reoccurring_values": None,
                      "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"},
                                       {"attr": "slope"}, {"attr": "stderr"}],
                      "cid_ce": [{"normalize": True}, {"normalize": False}],
                      "mean": None,
                      "benford_correlation": None,
                      "c3": [{"lag": lag} for lag in range(1, 4)],
                      "max_langevin_fixed_point": [{"m": 3, "r": 30}],
                      "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
                        "autocorrelation": [{"lag": lag} for lag in range(5)],
                        "percentage_of_reoccurring_values_to_all_values": None,
                       "absolute_sum_of_changes": None,
}


def extractFeaturesAndSave(col_name,patient_data_path,output_path):


    # set the current considered feature extraction
    features_to_be_extracted = current_considered_features

    #  name of the column for window ID's
    window_ID = "window_ID"

    #  Sorting (time) column
    time_col = "index"

    cols = [window_ID,time_col,col_name] # just use id,time and column to extract features
    df = pd.read_csv(patient_data_path,usecols=cols,nrows=1000) # nrows=1000

    for key in features_to_be_extracted.keys():
        feature_file_name = output_path + "/" +  col_name + "_" + key + ".csv"  # check if feature is already extracted

        if os.path.isfile(feature_file_name):
            continue  # do not extract again

        settings = {key: features_to_be_extracted[key]}
        print("Extracting: " + feature_file_name)
        start = time.time()

        # without stacked dataframe
        extracted_features = extract_features(df, column_id=window_ID, column_sort=time_col,
                                              default_fc_parameters=settings,n_jobs=0) # n_jobs=0
        end = time.time()
        print("Extracting: " + feature_file_name + " took" + str(end - start) + " seconds")


        """Impute the highly sparse features to get ready for training"""
        print("Start imputation")
        # replace NaN's with medians, -inf's with min value and +inf with max value
        # if no finite value exists, fill with zeros
        impute(extracted_features)
        print("End imputation")

        extracted_features.to_csv(feature_file_name)

        # remove and collect garbage
        del extracted_features
        gc.collect()



class NoDaemonProcess(multiprocessing.Process):
    """
    Customize the multiprocessing class
    """
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    """"""
    Process = NoDaemonProcess

def cli_main():
    # Get all the patient data, which was encoded and windowed in interim folder
    data_path = "../../data/interim/PPG_FieldStudy_Windowed/"
    subfiles = glob(data_path + "/*.csv")

    #  Make output path for saving the processed results
    output_path = "../../data/processed/PPG_FieldStudy_Extracted/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    #  Process each of the files and export to output path
    for patient_data_path in subfiles:
        """for each patient data, extract the features column by column
        so that they can be used as desired,seperately or altogether"""

        # Extract current subject from filename and make a dedicated directory
        current_subject = os.path.splitext(os.path.split(patient_data_path)[-1])[0]
        patient_output_path = os.path.join(output_path,current_subject)
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path)

        # find columns
        col_list = pd.read_csv(patient_data_path, nrows=2).columns
        col_list = list(col_list)[2:]
        if "Label" in col_list: col_list.remove("Label") # remove label from extraction


        # Create the partial function with pool and start extraction in parallel
        func = partial(extractFeaturesAndSave,patient_data_path=patient_data_path ,output_path= patient_output_path)

        pool = Pool() # MyPool()
        pool.map(func, col_list)
        pool.close()
        pool.join()

        #extractFeaturesAndSave(patient_data_path=patient_data_path,col_name=col_name,output_path= patient_output_path)

if __name__ == '__main__':
    cli_main()