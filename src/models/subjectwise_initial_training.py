import xgboost
import pandas as pd
import numpy as np
from itertools import product
import os
from glob import glob


# Features to perform the training on
optimal_features = {"maximum": None,
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


# Some helper functions


def checkIsInOptimals(optimal_features, filename):

    # get the exact feature name from filename
    feature_name = os.path.splitext(os.path.basename(filename))[0] # exact feature name
    for key in optimal_features.keys():
        if key in feature_name:
            return True
    return False



# Main training pipeline

def selectFeatures(extracted_features_path,patient_output_path):
    """takes all csv files in path, creates dataframe and performs a feature merging
    Args:
        extracted_features_path ([type]): [file with the extracted features]
    """


    """Continue regular selectFeatures"""
    path = extracted_features_path  # use your path
    all_files = glob(path + "/*.csv")


    # Early return of selected features
    if os.path.exists(patient_output_path + "/selected_features.csv"):
        print("Early returning of selected features !")

        df = pd.read_csv(path + "/selected_features.csv", index_col=0)
        if "labels" in df.columns:
            df = df.drop(['labels'], axis=1)

        return df


    li = []


    for filename in all_files:
        if "labels" in filename:  # do not append labels
            continue
        if "selected_features" in filename:
            continue
        if "png" in filename:
            continue

        if not checkIsInOptimals(optimal_features, filename):
            continue

        df = pd.read_csv(filename, index_col=0, header=0)
        li.append(df)

    df = pd.concat(li, axis=1, ignore_index=False)


    # Make sure no label exist
    if "labels" in df.columns:
        df = df.drop(['labels'], axis=1)

    # Save the selected features to relevant path
    df.to_csv(path + "/selected_features.csv")
    return df


def trainModel(patient_data, label_path, patient_output_path):

    pass


def evaluatePatient(patient_data_path,label_path,patient_output_path):
    """
    Evaluate the patient by merging features and training-outputting a model

    """


    # select and merge features and corresponding windows labels
    patient_data = selectFeatures(patient_data_path, patient_output_path)

    # Train a model and save the model and corresponding test results

    trainModel(patient_data,label_path,patient_output_path)

    pass


def cli_main():

    # set ID and sorting columns
    ID_column = "window_ID"
    sorting_column = "index"

    # Get all the patient data, which was encoded and windowed in interim folder
    data_path = "../../data/processed/PPG_FieldStudy_Extracted"
    subfiles =  [f.path for f in os.scandir(data_path) if f.is_dir()]

    #  Make output path for saving the selected results for training
    output_path =  "../../data/processed/PPG_FieldStudy_Selected"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Label path
    label_path =  "../../data/interim/PPG_FieldStudy_Windowed"


    #  Process each of the files and export to output path
    for patient_data_path in subfiles:
        """for each patient data, select the features column by column
        so that they can be used as desired,seperately or altogether"""

        if "labels" in patient_data_path:
            continue

        # Merge current subject data from filename and make a dedicated directory
        current_subject = os.path.splitext(os.path.split(patient_data_path)[-1])[0]
        patient_output_path = os.path.join(output_path, current_subject)
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path)


        evaluatePatient(patient_data_path,label_path,patient_output_path)



if __name__ == '__main__':

    cli_main()