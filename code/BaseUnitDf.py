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


class BaseUnitDf:
    def __init__(self, path_dir, trueBarInfo_path_dir):
        self.__df = pd.read_pickle(path_dir + "/" + os.path.basename(path_dir) + ".pkl")
        self.__df = self.__df.drop(
            [
                "pose:x_15",
                "pose:y_15",
                "pose:x_16",
                "pose:y_16",
                "pose:x_17",
                "pose:y_17",
                "pose:x_18",
                "pose:y_18",
                "pose:x_22",
                "pose:y_22",
                "pose:x_23",
                "pose:y_23",
                "pose:x_24",
                "pose:y_24",
                "pose:x_19",
                "pose:y_19",
                "pose:x_21",
                "pose:y_21",
                "pose:x_20",
                "pose:y_20",
                "frameTimeDiffPer1Unit",
            ],
            axis=1,
        )
        self.__df.name = os.path.basename(path_dir)

        self.__barInfo = np.loadtxt(
            path_dir + "/" + os.path.basename(path_dir) + "BarInfo.csv",
            delimiter=",",
            dtype=np.int32,
        )
        self.__trueBarInfo = np.loadtxt(
            trueBarInfo_path_dir + ".csv", delimiter=",", dtype=np.int32
        )

        self.__unitSize = self.setUnitSize()
        self.__stepDiff = 0
        self.__samplingFrameSize = 0

    def makeUnitRow(self, numOfSamplingNumIn1unit=2):
        self.__stepDiff, self.__samplingFrameSize = self.setSamplingSize(
            numOfSamplingNumIn1unit
        )
        return self.samplingFromUnit()

    def samplingFromUnit(self):
        totalUnitLength = len(self.__trueBarInfo)
        unitColumnNames = self.makeColumnName()
        unitDf = pd.DataFrame(columns=unitColumnNames)
        unitDf.name = self.__df.name

        for unit in range(totalUnitLength):
            selectedUnitDf = self.__df.iloc[
                self.__barInfo[:unit].sum() : self.__barInfo[: unit + 1].sum(), :
            ]
            for step in range(self.__stepDiff):
                tempDf = selectedUnitDf.iloc[
                    step : self.__unitSize : self.__stepDiff, :
                ]
                tempDf = np.append(
                    tempDf.to_numpy().flatten(), [int(self.__trueBarInfo[unit])], axis=0
                )
                if tempDf.shape[0] != 1021:
                    break
                tempDf = pd.DataFrame(
                    tempDf.reshape(-1, tempDf.shape[0]), columns=unitColumnNames
                )
                unitDf = unitDf.append(tempDf, ignore_index=True)
        return unitDf

    def makeColumnName(self):
        unitColumns = np.array([])
        for i in range(self.__samplingFrameSize):
            unitColumns = np.concatenate(
                (unitColumns, np.array(self.__df.columns)), axis=0
            )
        unitColumns = np.append(unitColumns, ["label"], axis=0)
        return unitColumns

    def setSamplingSize(self, numOfSamplingNumIn1unit):
        if self.isPossibleSamplingSize(numOfSamplingNumIn1unit) == True:
            self.__samplingFrameSize = int(self.__unitSize / numOfSamplingNumIn1unit)
            self.__stepDiff = numOfSamplingNumIn1unit
            return self.__stepDiff, self.__samplingFrameSize

    def isPossibleSamplingSize(self, numOfSamplingNumIn1unit):
        print("\n작업 파일 이름 : {}".format(self.__df.name))
        print("현재 Unit size는 {}입니다.".format(self.__unitSize))
        if self.__unitSize % numOfSamplingNumIn1unit == 0:
            print("한 마디에서 {}번의 마디가 생성됩니다.".format(numOfSamplingNumIn1unit))
            print(
                "한 마디에서 추출되어 발생할 표본의 크기는 {}개입니다.".format(
                    int(self.__unitSize / numOfSamplingNumIn1unit), end="\n"
                )
            )
            return True
        else:
            print(
                "불가능한 sampling Number입니다. unitSize % numOfSamplingNumIn1unit = {}".format(
                    self.__unitSize % numOfSamplingNumIn1unit
                )
            )
        return False

    def setUnitSize(self):
        if (self.__barInfo.max() - 1) % 2 == 0:
            return self.__barInfo.max() - 1
        else:
            return self.__barInfo.max() - 2

    def __str__(self):
        print("현재 데이터 셋 이름 : {}".format(self.__df.name))
        print("현재 unitSize : {}".format(self.__unitSize))
        print("현재 stepDiff : {}".format(self.__stepDiff))
        print("현재 samplingFrameSize : {}".format(self.__samplingFrameSize), end="\n\n")

