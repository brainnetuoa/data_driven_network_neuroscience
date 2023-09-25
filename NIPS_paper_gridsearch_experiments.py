"""
Code to run the experiments based on Grid Search

Under INPUT PARAMETERS, select datasets/parcellations/models to run on

Results will be generated as .csv files in the same filepath
"""

from scipy.io import loadmat

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


"""
INPUT PARAMETERS -
Define the datasets, parcellations, and models for this run of Grid Search
"""

dataset = ["abide", "adni", "ppmi", "taowu", "neurocon"]
parcellation = ["AAL116", "harvard48", "schaefer100", "kmeans100", "ward100"]
models = ["LR", "SVC", "kNN", "RF"]


""" assign ROI counts to specified parcellation schemes """
num_roi = []
for x in range(0, len(parcellation)):
    if parcellation[x] == "AAL116":
        num_roi.append(116)
    elif parcellation[x] == "harvard48":
        num_roi.append(48)
    elif parcellation[x] == "schaefer100":
        num_roi.append(100)
    elif parcellation[x] == "kmeans100":
        num_roi.append(100)
    elif parcellation[x] == "ward100":
        num_roi.append(100)


def load_data(dataset, parcellation, num_rois):
    """
    Return [X, y] by first loading data then creating feature matrices by
    loading edge weights into X, and loading class labels into y
    """
    subject_paths = glob.glob("datasets/" + dataset + "/*")
    subject_names = [os.path.basename(x) for x in subject_paths]
    y = []
    X = np.zeros(num_rois * num_rois)

    for x in range(0, len(subject_paths)):
        mat = loadmat(
            subject_paths[x]
            + "/"
            + subject_names[x]
            + "_"
            + parcellation
            + "_correlation_matrix.mat"
        )
        adjacency = mat["data"]

        X = np.vstack([X, adjacency.flatten()])

        # generate class label list y based on subject ID
        if "control" in subject_names[x]:
            y.append(1)
        elif "patient" in subject_names[x]:
            y.append(2)
        elif "mci" or "prodromal" in subject_names[x]:
            y.append(3)
        elif "emci" or "swedd" in subject_names[x]:
            y.append(4)
        elif "SMC" in subject_names[x]:
            y.append(5)
        elif "LMCI" in subject_names[x]:
            y.append(6)

    X = np.delete(X, 0, axis=0)  # delete empty first row of zeros from X

    return [X, y]


def cross_validation(X, y, dataset, parcellation, modelname):
    """
    Runs cross-validation and saves the Grid Search results
    into a .csv in the same filepath
    """
    if modelname == "LR":
        parameters = {"penalty": ("l2", "none")}
        model = LogisticRegression(max_iter=1000000)
    elif modelname == "kNN":
        parameters = {
            "n_neighbors": (3, 4, 5, 6),
            "weights": ("uniform", "distance"),
            "p": (1, 2),
        }
        model = KNeighborsClassifier()
    elif modelname == "SVC":
        parameters = {
            "kernel": ("rbf", "linear", "poly", "sigmoid"),
            "C": [0.1, 1, 10],
            "gamma": ("auto", "scale"),
        }
        model = SVC()
    elif modelname == "RF":
        parameters = {
            "n_estimators": (50, 100, 150, 200),
            "criterion": ("gini", "entropy"),
            "max_depth": (2, 3, 4, 5),
        }
        model = RandomForestClassifier()
    elif modelname == "NB":
        parameters = {}
        model = GaussianNB()

    clf = GridSearchCV(model, parameters)
    clf.fit(X, y)

    df = pd.DataFrame.from_dict(clf.cv_results_)
    filepath = Path(dataset + "_" + parcellation + "_" + modelname + ".csv")
    df.to_csv(filepath)

    print("------------------------------")
    print("dataset:", dataset)
    print("parcellation:", parcellation)
    print("model:", modelname)
    print("DONE")
    print("------------------------------")


""" runs the loop and call methods to run experiments """
for i in range(0, len(dataset)):
    for j in range(0, len(parcellation)):
        for k in range(0, len(models)):
            data = load_data(dataset[i], parcellation[j], num_roi[j])
            cross_validation(data[0], data[1], dataset[i], parcellation[j], models[k])
