from sklearn.cluster import DBSCAN
import os
import json
import numpy as np
import pandas as pd
import random as rd
from collections import Counter
import matplotlib.pyplot as plt
from itertools import combinations
from pprint import pprint as pp
import copy
from BaseUnitDf import *
from DataAugmentation import *
from mergeAllDf import *


def makeAllSiftedDfList(basicUnitDf, iterNum=1, mu=0, sigma=0.1):
    DataAugmentationObject = DataAugmentation(basicUnitDf)
    siftedDfList = [basicUnitDf]
    for i in range(5, len(DataAugmentationObject.getFuncList()) + 1):
        siftedDfList.append(
            DataAugmentationObject.applySiftFuncWithIteration(i, iterNum, mu, sigma)
        )
    return siftedDfList


def mergeAllDf(siftedList):
    allSiftedDf = pd.DataFrame(columns=siftedList[0].columns)
    for i in range(len(siftedList)):
        allSiftedDf = allSiftedDf.append(siftedList[i], ignore_index=True)
    return allSiftedDf


def makeBasicBarUnitDf(path_dir, trueBarInfo_path_dir):
    if os.path.exists(
        "../data/4_data_Augmentation/" + os.path.basename(path_dir) + "Base" + ".pkl"
    ):
        print(
            "{}의 data augmentation이 끝났습니다. 종료합니다.".format(
                os.path.basename(path_dir) + "Base"
            )
        )
    else:
        print(
            "{}의 data augmentation이 수행되지 않았습니다. 작업을 시작합니다.".format(
                os.path.basename(path_dir) + "Base"
            )
        )
        basicBarUnitObject = BaseUnitDf(path_dir, trueBarInfo_path_dir)
        basicBarUnitDf = basicBarUnitObject.makeUnitRow()
        basicBarUnitObject.__str__()
        basicBarUnitDf.to_pickle(
            "../data/4_data_Augmentation/"
            + os.path.basename(path_dir)
            + "Base"
            + ".pkl"
        )


def makeAugmentaedDf_fromBasicBarUnit(path_dir):
    if os.path.exists(
        "../data/4_data_Augmentation/"
        + os.path.basename(path_dir)
        + "Augmentated"
        + ".pkl"
    ):
        print(
            "{}의 data augmentation이 끝났습니다. 종료합니다.".format(
                os.path.basename(path_dir) + "Augmentated"
            )
        )
    else:
        print(
            "{}의 data augmentation이 수행되지 않았습니다. 작업을 시작합니다.".format(
                os.path.basename(path_dir) + "Augmentated"
            )
        )
        print("현재 파일 : {}".format(os.path.basename(path_dir)))
        basicBarUnitDf = pd.read_pickle(
            "../data/4_data_Augmentation/"
            + os.path.basename(path_dir)
            + "Base"
            + ".pkl"
        )
        siftedDfList = makeAllSiftedDfList(
            basicBarUnitDf, iterNum=1, mu=-0.5, sigma=0.2
        )
        allSiftedDf = mergeAllDf(siftedDfList)
        allSiftedDf.to_pickle(
            "../data/4_data_Augmentation/"
            + os.path.basename(path_dir)
            + "Augmentated"
            + ".pkl"
        )
