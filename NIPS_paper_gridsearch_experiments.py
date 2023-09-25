"""
Code to run the experiments based on Grid Search

Under INPUT PARAMETERS, select datasets/parcellations/models to run on

Results will be generated as .csv files in the same filepath
"""
import warnings
from scipy.io import loadmat
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
import random
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

proportion = ['9fold', '8fold', '7fold', '6fold', '5fold', '4fold', '3fold']
dataset = ["matai", "abide", "adni", "ppmi", "taowu", "neurocon"]
parcellation = ["AAL116", "harvard48", "schaefer100", "kmeans100", "ward100"]
models = ["LR", "NB", "SVC", "kNN", "RF"]

seed = 41
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)


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


def load_data(dataset, parcellation, num_rois, proportion=''):
    """
    Return [X, y] by first loading data then creating feature matrices by
    loading edge weights into X, and loading class labels into y
    """
    print('Loading data...')
    group_paths = glob.glob("data/" + dataset + "_" + parcellation.lower() + "/*")
    subject_paths = []
    splits = {}
    for split in ['train', 'val', 'test']:
        if proportion:
            f = open("data/" + dataset + "_schaefer100_" + proportion + "/" + split + ".index", 'r')
        else:
            f = open("data/" + dataset + "_schaefer100/" + split + ".index", 'r')
        splits[split] = [[int(s) for s in l.split(',')] for l in f.readlines()]

    for path in group_paths:
        if path[-6:] == '.index':
            continue
        subject_paths += glob.glob(path + '/*')
    y = []
    X = np.zeros(num_rois * num_rois)

    for x in range(0, len(subject_paths)):
        adjacency = np.loadtxt(subject_paths[x], delimiter=' ')

        if adjacency.shape[0] < num_rois:
            print('A subjuct contains 99 ROIs.')
            newrow = [0.0] * adjacency.shape[0]
            adjacency = np.vstack([adjacency, newrow])
            newrow = [0.0] * num_rois
            adjacency = np.vstack([adjacency.T, newrow]).T
        X = np.vstack([X, adjacency.flatten()])

        # generate class label list y based on subject ID
        if "control" in subject_paths[x] or "baseline" in subject_paths[x]:
            y.append(1)
        elif "patient" in subject_paths[x] or "postseason" in subject_paths[x]:
            y.append(2)
        elif "emci" in subject_paths[x] or "swedd" in subject_paths[x]:
            y.append(3)
        elif "mci" in subject_paths[x] or "prodromal" in subject_paths[x]:
            y.append(4)
        elif "SMC" in subject_paths[x]:
            y.append(5)
        elif "LMCI" in subject_paths[x]:
            y.append(6)

    X = np.delete(X, 0, axis=0)  # delete empty first row of zeros from X

    split_iters = []
    if splits:
        for fold in range(len(splits['train'])):
            split_iters.append((splits['train'][fold], splits['val'][fold]))

    return [X, y], splits, split_iters


def cross_validation(X, y, dataset, parcellation, modelname, splits, split_iters, proportion=''):
    """
    Runs cross-validation and saves the Grid Search results
    into a .csv in the same filepath
    """
    if modelname == "LR":
        # parameters = {"penalty": ("l2", 'l1')}
        parameters = {}
        model = LogisticRegression(max_iter=1000000, random_state=seed, multi_class='multinomial', solver='saga')
    elif modelname == "kNN":
        parameters = {
            #"n_neighbors": (3, 4, 5, 6),
            "weights": ("uniform", "distance"),
            "p": (1, 2),
        }
        model = KNeighborsClassifier()
    elif modelname == "SVC":
        parameters = {
            "kernel": ("rbf", "linear", "poly", "sigmoid"),
            # "C": [0.1, 1, 10],
            # "gamma": ("auto", "scale"),
        }
        model = SVC(random_state=seed)
    elif modelname == "RF":
        parameters = {
            "n_estimators": (50, 100, 150, 200),
            # "criterion": ("gini", "entropy"),
            # "max_depth": (2, 3, 4, 5),
        }
        model = RandomForestClassifier(random_state=seed)
    elif modelname == "NB":
        parameters = {}
        model = GaussianNB()
    else:
        raise NotImplementedError

    clf = GridSearchCV(model, parameters, cv=split_iters, refit=False) if split_iters else GridSearchCV(model, parameters)
    clf.fit(X, y)
    best_params = clf.best_params_

    df = pd.DataFrame.from_dict(clf.cv_results_)
    if proportion:
        filepath = Path('result/' + dataset + "_" + parcellation + "_" + proportion + "_" + modelname + ".csv")
    else:
        filepath = Path('result/' + dataset + "_" + parcellation + "_" + modelname + ".csv")
    df.to_csv(filepath)

    print("------------------------------")
    print("dataset:", dataset)
    print('subject number:', len(y))
    print("parcellation:", parcellation)
    print("model:", modelname)
    print("DONE")
    if splits:
        accs = []
        for i, test_idx in enumerate(splits['test']):
            train_idx = splits['train'][i]
            train_X = X[train_idx]
            train_y = np.array(y)[train_idx]
            fold_clf = model.set_params(**best_params)
            fold_clf.fit(train_X, train_y)
            predicted = fold_clf.predict(X[test_idx])
            test_acc = sum(predicted == np.array(y)[test_idx]) / len(np.array(y)[test_idx])
            accs.append(test_acc)
        acc_mean = np.mean(np.array(accs))
        acc_std = np.std(np.array(accs))
        print('test mean acc: {}'.format(acc_mean))
        print('test acc std: {}'.format(acc_std))
    print("------------------------------")


""" runs the loop and call methods to run experiments """
warnings.filterwarnings("ignore")
for p in proportion:
    for i in range(0, len(dataset)):
        for j in range(0, len(parcellation)):
            for k in range(0, len(models)):
                data, splits, split_iters = load_data(dataset[i], parcellation[j], num_roi[j], proportion=p)
                cross_validation(data[0], data[1], dataset[i], parcellation[j], models[k], splits, split_iters, proportion=p)
