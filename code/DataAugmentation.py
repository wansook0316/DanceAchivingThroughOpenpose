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
import random as rd


class DataAugmentation:
    def __init__(self, baseDf):
        self.__df = copy.deepcopy(baseDf)
        self.__baseDf = copy.deepcopy(baseDf)
        self.__funcList = np.array(
            [
                "siftingWholeBody",
                "siftingUpperBody0",
                "siftingUpperBody1",
                "siftingUpperBody2_5",
                "siftingUpperBody3_6",
                "siftingUpperBody4_8",
                "siftingLowerBody9_12",
                "siftingLowerBody10_13",
                "siftingLowerBody11_14",
            ]
        )

    def resetDf(self):
        print("====== 기존 작업 DataFrame 재설정 ======")
        self.__df = copy.deepcopy(self.__baseDf)

    def findSelectColumnName(self, keyPointNum):
        if keyPointNum == -1:
            keyPointList = list(map(lambda i: i, range(15)))
            result = list(
                map(
                    lambda keyPointNum: [
                        "pose:x_" + str(keyPointNum),
                        "pose:y_" + str(keyPointNum),
                    ],
                    keyPointList,
                )
            )
            return np.array(result).flatten()

        elif keyPointNum == 0:
            keyPointList = [0]
            result = list(
                map(
                    lambda keyPointNum: [
                        "pose:x_" + str(keyPointNum),
                        "pose:y_" + str(keyPointNum),
                    ],
                    keyPointList,
                )
            )
            return np.array(result).flatten()

        elif keyPointNum == 1:
            keyPointList = [1, 2, 3, 4, 5, 6, 7]
            result = list(
                map(
                    lambda keyPointNum: [
                        "pose:x_" + str(keyPointNum),
                        "pose:y_" + str(keyPointNum),
                    ],
                    keyPointList,
                )
            )
            return np.array(result).flatten()

        elif keyPointNum == 2:
            rightKeyPointList = [2, 3, 4]
            leftKeyPointList = [5, 6, 7]

        elif keyPointNum == 3:
            rightKeyPointList = [3, 4]
            leftKeyPointList = [6, 7]

        elif keyPointNum == 4:
            rightKeyPointList = [4]
            leftKeyPointList = [7]

        elif keyPointNum == 9:
            rightKeyPointList = [9, 10, 11]
            leftKeyPointList = [12, 13, 14]

        elif keyPointNum == 10:
            rightKeyPointList = [10, 11]
            leftKeyPointList = [13, 14]

        elif keyPointNum == 11:
            rightKeyPointList = [11]
            leftKeyPointList = [14]

        rightColumnName_X = list(
            map(lambda keyPointNum: "pose:x_" + str(keyPointNum), rightKeyPointList)
        )
        rightColumnName_Y = list(
            map(lambda keyPointNum: "pose:y_" + str(keyPointNum), rightKeyPointList)
        )
        leftColumnName_X = list(
            map(lambda keyPointNum: "pose:x_" + str(keyPointNum), leftKeyPointList)
        )
        leftColumnName_Y = list(
            map(lambda keyPointNum: "pose:y_" + str(keyPointNum), leftKeyPointList)
        )
        return (
            np.array(rightColumnName_X),
            np.array(rightColumnName_Y),
            np.array(leftColumnName_X),
            np.array(leftColumnName_Y),
        )

    def generateNoise(self, mu, sigma):
        # rd.random.seed()
        # rightNoise_X = np.random.normal(mu, sigma, 1)
        # rightNoise_Y = np.random.normal(mu, sigma, 1)
        # leftNoise_X = rightNoise_X
        # leftNoise_Y = rightNoise_Y
        rightNoise_X = (mu + rd.random()) * sigma
        rightNoise_Y = (mu + rd.random()) * sigma
        leftNoise_X = (mu + rd.random()) * sigma
        leftNoise_Y = (mu + rd.random()) * sigma
        print(rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y)
        return rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y

    def updateDf(
        self,
        rightList_X,
        rightList_Y,
        leftList_X,
        leftList_Y,
        rightNoise_X,
        rightNoise_Y,
        leftNoise_X,
        leftNoise_Y,
    ):
        for i in range(self.__df.shape[0]):
            self.__df.loc[i : i + 1, rightList_X] = self.__df.loc[
                i : i + 1, rightList_X
            ].apply(lambda x: x + rightNoise_X, axis=1)
            self.__df.loc[i : i + 1, rightList_Y] = self.__df.loc[
                i : i + 1, rightList_Y
            ].apply(lambda x: x + rightNoise_Y, axis=1)
            self.__df.loc[i : i + 1, leftList_X] = self.__df.loc[
                i : i + 1, leftList_X
            ].apply(lambda x: x + leftNoise_X, axis=1)
            self.__df.loc[i : i + 1, leftList_Y] = self.__df.loc[
                i : i + 1, leftList_Y
            ].apply(lambda x: x + leftNoise_Y, axis=1)

    def siftingWholeBody(self, mu, sigma):
        keyPointList = self.findSelectColumnName(-1)
        for i in range(self.__df.shape[0]):
            self.__df.loc[i : i + 1, keyPointList] = self.__df.loc[
                i : i + 1, keyPointList
            ].apply(lambda x: x + np.random.normal(mu, sigma, 1), axis=1)

    def siftingUpperBody0(self, mu, sigma):
        keyPointList = self.findSelectColumnName(0)
        for i in range(self.__df.shape[0]):
            self.__df.loc[i : i + 1, keyPointList] = self.__df.loc[
                i : i + 1, keyPointList
            ].apply(lambda x: x + np.random.normal(mu, sigma, 1), axis=1)

    def siftingUpperBody1(self, mu, sigma):
        keyPointList = self.findSelectColumnName(1)
        for i in range(self.__df.shape[0]):
            self.__df.loc[i : i + 1, keyPointList] = self.__df.loc[
                i : i + 1, keyPointList
            ].apply(lambda x: x + np.random.normal(mu, sigma, 1), axis=1)

    def siftingUpperBody2_5(self, mu, sigma):
        rightHandList_X, rightHandList_Y, leftHandList_X, leftHandList_Y = self.findSelectColumnName(
            2
        )
        rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y = self.generateNoise(
            mu, sigma
        )
        if np.sign(leftNoise_Y) == np.sign(rightNoise_X):
            leftNoise_Y = -rightNoise_Y
        self.updateDf(
            rightHandList_X,
            rightHandList_Y,
            leftHandList_X,
            leftHandList_Y,
            rightNoise_X,
            rightNoise_Y,
            leftNoise_X,
            leftNoise_Y,
        )

    def siftingUpperBody3_6(self, mu, sigma):
        rightHandList_X, rightHandList_Y, leftHandList_X, leftHandList_Y = self.findSelectColumnName(
            3
        )
        rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y = self.generateNoise(
            mu, sigma
        )
        self.updateDf(
            rightHandList_X,
            rightHandList_Y,
            leftHandList_X,
            leftHandList_Y,
            rightNoise_X,
            rightNoise_Y,
            leftNoise_X,
            leftNoise_Y,
        )

    def siftingUpperBody4_8(self, mu, sigma):
        rightHandList_X, rightHandList_Y, leftHandList_X, leftHandList_Y = self.findSelectColumnName(
            4
        )
        rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y = self.generateNoise(
            mu, sigma
        )
        self.updateDf(
            rightHandList_X,
            rightHandList_Y,
            leftHandList_X,
            leftHandList_Y,
            rightNoise_X,
            rightNoise_Y,
            leftNoise_X,
            leftNoise_Y,
        )

    def siftingLowerBody9_12(self, mu, sigma):
        rightLegList_X, rightLegList_Y, leftLegList_X, leftLegList_Y = self.findSelectColumnName(
            9
        )
        rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y = self.generateNoise(
            mu, sigma
        )
        if np.sign(leftNoise_Y) == np.sign(rightNoise_X):
            leftNoise_Y = -rightNoise_Y
        self.updateDf(
            rightLegList_X,
            rightLegList_Y,
            leftLegList_X,
            leftLegList_Y,
            rightNoise_X,
            rightNoise_Y,
            leftNoise_X,
            leftNoise_Y,
        )

    def siftingLowerBody10_13(self, mu, sigma):
        rightLegList_X, rightLegList_Y, leftLegList_X, leftLegList_Y = self.findSelectColumnName(
            10
        )
        rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y = self.generateNoise(
            mu, sigma
        )
        self.updateDf(
            rightLegList_X,
            rightLegList_Y,
            leftLegList_X,
            leftLegList_Y,
            rightNoise_X,
            rightNoise_Y,
            leftNoise_X,
            leftNoise_Y,
        )

    def siftingLowerBody11_14(self, mu, sigma):
        rightLegList_X, rightLegList_Y, leftLegList_X, leftLegList_Y = self.findSelectColumnName(
            11
        )
        rightNoise_X, rightNoise_Y, leftNoise_X, leftNoise_Y = self.generateNoise(
            mu, sigma
        )
        self.updateDf(
            rightLegList_X,
            rightLegList_Y,
            leftLegList_X,
            leftLegList_Y,
            rightNoise_X,
            rightNoise_Y,
            leftNoise_X,
            leftNoise_Y,
        )

    def getSiftedDf(self):
        return self.__df

    def getFuncList(self):
        return self.__funcList

    def applySiftFuncWithIteration(self, selectNum, iterNum=1, mu=0, sigma=0.1):
        SiftedDf = pd.DataFrame(columns=self.__baseDf.columns)
        combinationList = list(combinations(self.__funcList, selectNum))
        print(
            "\n********************* {}C{} 작업 시작 **********************".format(
                len(self.__funcList), selectNum
            )
        )
        print(
            "{}개의 작업을 선택했을 때 발생하는 경우의 수는 {}입니다.".format(selectNum, len(combinationList))
        )
        print("해당 작업을 {} 번 반복합니다.".format(iterNum), end="\n\n")
        for i in range(iterNum):
            for j in range(len(combinationList)):
                self.resetDf()
                print(self.__df.tail())
                print("{} 중 {}번째, 진행중 ...".format(len(combinationList), j + 1))
                # print(combinationList[j])
                for k in range(selectNum):
                    print("{} 작업 진행중 ...".format(combinationList[j][k]))
                    getattr(self, combinationList[j][k])(mu, sigma)

                SiftedDf = SiftedDf.append(self.__df, ignore_index=True)
                # print(SiftedDf.describe())
                # self.resetDf()
        return SiftedDf
