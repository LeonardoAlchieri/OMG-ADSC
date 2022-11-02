from __future__ import print_function

import argparse
import csv
import os
import sys
from typing import Tuple

import torch
import numpy
import pandas
from scipy.stats import pearsonr


def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(y_true, y_pred)


def f1(y_true, y_pred):
    from sklearn.metrics import f1_score

    label = [0, 1, 2, 3, 4, 5, 6]
    return f1_score(y_true, y_pred, labels=label, average="micro")


def ccc(input, target):
    # I have to force to tensors
    input = torch.Tensor(input)
    target = torch.Tensor(target)

    vx = input - torch.mean(input)
    vy = target - torch.mean(target)
    rho = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(torch.pow(vx, 2)))
        * torch.sqrt(torch.sum(torch.pow(vy, 2)))
    )
    x_m = torch.mean(input)
    y_m = torch.mean(target)
    x_s = torch.std(input)
    y_s = torch.std(target)
    ccc = (
        2
        * rho
        * x_s
        * y_s
        / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    )
    return ccc


def calculateCCC(validationFile, modelOutputFile) -> Tuple[float, float]:

    dataY = pandas.read_csv(validationFile, header=0, sep=",")

    dataYPred = pandas.read_csv(modelOutputFile, header=0, sep=",")

    dataYArousal = dataY["arousal"]
    dataYValence = dataY["valence"]
    dataYPredArousal = dataYPred["arousal"]
    dataYPredValence = dataYPred["valence"]

    arousalCCC = ccc(dataYArousal, dataYPredArousal)
    arousalmse = mse(dataYArousal, dataYPredArousal)
    valenceCCC = ccc(dataYValence, dataYPredValence)
    valencemse = mse(dataYValence, dataYPredValence)

    print("Arousal CCC: ", arousalCCC)
    print("Arousal MSE: ", arousalmse)
    print("Valence CCC: ", valenceCCC)
    print("Valence MSE: ", valencemse)
    return arousalCCC, valenceCCC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("validationFile")
    parser.add_argument("modelOutputFile")

    opt = parser.parse_args()
    if not os.path.exists(opt.validationFile):
        print("Cannot find validation File")
        sys.exit(-1)

    if not os.path.exists(opt.modelOutputFile):
        print("Cannot find modelOutput File")
        sys.exit(-1)

    calculateCCC(opt.validationFile, opt.modelOutputFile)
