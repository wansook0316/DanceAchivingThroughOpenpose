import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial


def setPoseName(df, axis, keyPointNum):
    columnList = df.columns
    if not (axis == "x" or axis == "y" or axis == "X" or axis == "Y"):
        print("없는 pose axis를 설정하셨습니다. 잘못된 입력입니다.")
        return -1
    if keyPointNum < 0 or keyPointNum > 25:
        print("없는 pose_num을 설정하셨습니다. 잘못된 입력입니다.")
        return -1
    colName = "pose:" + str(axis) + "_" + str(keyPointNum)
    return colName


def setPoseNameXY(df, keyPointNum):
    return setPoseName(df, "x", keyPointNum), setPoseName(df, "y", keyPointNum)


def get1UnitDf(df, numOf1Unit, barInfoList):
    if numOf1Unit <= 0 or numOf1Unit > len(barInfoList):
        print("잘못된 입력입니다. 해당 마디는 정보에 없습니다.")
        return -1
    barLength = barInfoList[numOf1Unit - 1]
    prevBarLength = barInfoList[0 : numOf1Unit - 1].sum()
    return df[prevBarLength : prevBarLength + barLength]


def plot3dPosePer1Unit(df, keyPointNum, unitNum, barInfoList):
    poseNameOfx, poseNameOfy = setPoseNameXY(df, keyPointNum)
    currentUnitDf = get1UnitDf(df, unitNum, barInfoList)
    x = currentUnitDf[poseNameOfx].to_numpy()
    y = currentUnitDf[poseNameOfy].to_numpy()
    z = currentUnitDf["frameTimeDiffPer1Unit"].to_numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.view_init(10, 90)
    ax.plot(x, y, z, alpha=0.6, marker="o")
    plt.title("{}th Unit Plot Infomation In {} Key Point".format(unitNum, keyPointNum))
    plt.show()


def setPoseVectorXY(df, keyPointNum, unitNum, barInfo):
    xName, yName = setPoseNameXY(df, keyPointNum)
    currentDf = get1UnitDf(df, unitNum, barInfo)
    return currentDf[[xName, yName]].to_numpy()


def setTwoMatSizeEqual(mat1, mat2):
    frameSize = np.shape(mat1)[0]
    if np.shape(mat1)[0] != np.shape(mat2)[0]:
        if np.shape(mat1)[0] > np.shape(mat2)[0]:
            frameSize = np.shape(mat1)[0] - 1
        else:
            frameSize = np.shape(mat2)[0] - 1
    return mat1[0:frameSize, :], mat2[0:frameSize, :]


def cosSimilarity(mat1, mat2):
    mat1Flat = np.hstack(mat1)
    mat2Flat = np.hstack(mat2)
    return 1 - spatial.distance.cosine(mat1Flat, mat2Flat)


def compare2SongUnitBy3dPlot(
    df1, df2, KeyPointNum, unitNum, barInfoList1, barInfoList2, zAngle=10, xAngle=45
):
    poseNameOfx, poseNameOfy = setPoseNameXY(df1, KeyPointNum)
    currentUnitDf1 = get1UnitDf(df1, unitNum, barInfoList1)
    currentUnitDf2 = get1UnitDf(df2, unitNum, barInfoList2)

    mat1 = currentUnitDf1.to_numpy()
    mat2 = currentUnitDf2.to_numpy()
    mat1, mat2 = setTwoMatSizeEqual(mat1, mat2)
    print(np.shape(mat1), np.shape(mat2))
    cosSim = cosSimilarity(mat1, mat2)

    x1 = currentUnitDf1[poseNameOfx].to_numpy()
    y1 = currentUnitDf1[poseNameOfy].to_numpy()
    z1 = currentUnitDf1["frameTimeDiffPer1Unit"].to_numpy()
    x2 = currentUnitDf2[poseNameOfx].to_numpy()
    y2 = currentUnitDf2[poseNameOfy].to_numpy()
    z2 = currentUnitDf2["frameTimeDiffPer1Unit"].to_numpy()

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(
        "{}th Unit Plot Infomation Between {} & {} Song In {} Key Point".format(
            unitNum, df1.name, df2.name, KeyPointNum
        ),
        fontsize=16,
    )
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.xlabel("x Value Of Width Pixel")
    plt.ylabel("y Value Of Width Pixel")
    ax.view_init(zAngle, xAngle)

    ax.plot(
        x1,
        y1,
        z1,
        alpha=0.5,
        marker="o",
        color="green",
        label="{} Song".format(df1.name),
    )
    ax.plot(
        x2,
        y2,
        z2,
        alpha=0.5,
        marker="o",
        color="blue",
        label="{} Song".format(df2.name),
    )
    ax.legend()

    plt.title("Cos similaritiy is {}%".format(round(cosSim * 100, 3)))
    plt.show()


def compareUnitBy3dPlot(
    df, KeyPointNum, unitNum1, unitNum2, barInfoList, zAngle=10, xAngle=45
):
    poseNameOfx, poseNameOfy = setPoseNameXY(df, KeyPointNum)
    currentUnitDf1 = get1UnitDf(df, unitNum1, barInfoList)
    currentUnitDf2 = get1UnitDf(df, unitNum2, barInfoList)

    mat1 = currentUnitDf1.to_numpy()
    mat2 = currentUnitDf2.to_numpy()
    mat1, mat2 = setTwoMatSizeEqual(mat1, mat2)
    cosSim = cosSimilarity(mat1, mat2)

    x1 = currentUnitDf1[poseNameOfx].to_numpy()
    y1 = currentUnitDf1[poseNameOfy].to_numpy()
    z1 = currentUnitDf1["frameTimeDiffPer1Unit"].to_numpy()
    x2 = currentUnitDf2[poseNameOfx].to_numpy()
    y2 = currentUnitDf2[poseNameOfy].to_numpy()
    z2 = currentUnitDf2["frameTimeDiffPer1Unit"].to_numpy()

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(
        "{}th & {}th Unit Plot Infomation In {} Key Point".format(
            unitNum1, unitNum2, KeyPointNum
        ),
        fontsize=16,
    )
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.xlabel("x Value Of Width Pixel")
    plt.ylabel("y Value Of Width Pixel")
    ax.view_init(zAngle, xAngle)

    ax.plot(
        x1,
        y1,
        z1,
        alpha=0.5,
        marker="o",
        color="green",
        label="{}th Unit".format(unitNum1),
    )
    ax.plot(
        x2,
        y2,
        z2,
        alpha=0.5,
        marker="o",
        color="blue",
        label="{}th Unit".format(unitNum2),
    )
    ax.legend()

    plt.title("Cos similaritiy is {}%".format(round(cosSim * 100, 3)))
    plt.show()


if __name__ == "__main__":
    path_dir1 = "../data/3_dataFrame/30sm1"
    path_dir2 = "../data/3_dataFrame/30sm2"
    path_dir3 = "../data/3_dataFrame/30sm3"
    path_dir4 = "../data/3_dataFrame/30sm4"
    path_dir5 = "../data/3_dataFrame/30sm5"
    path_dir6 = "../data/3_dataFrame/30sm6"

    path_dir8 = "../data/3_dataFrame/30sm8"
    path_dir9 = "../data/3_dataFrame/30sm9"
    path_dir10 = "../data/3_dataFrame/30sm10"
    path_dir11 = "../data/3_dataFrame/30sm11"
    path_dir12 = "../data/3_dataFrame/30sm12"
    path_dir13 = "../data/3_dataFrame/30sm13"
    path_dir14 = "../data/3_dataFrame/30sm14"
    path_dir15 = "../data/3_dataFrame/30sm15"

    barInfo1 = np.loadtxt(
        path_dir1 + "/" + os.path.basename(path_dir1) + "BarInfo.csv",
        delimiter=",",
        dtype=np.int32,
    )
    df1 = pd.read_pickle(path_dir1 + "/" + os.path.basename(path_dir1) + ".pkl")
    df1.name = os.path.basename(path_dir1)

    barInfo2 = np.loadtxt(
        path_dir2 + "/" + os.path.basename(path_dir2) + "BarInfo.csv",
        delimiter=",",
        dtype=np.int32,
    )
    df2 = pd.read_pickle(path_dir2 + "/" + os.path.basename(path_dir2) + ".pkl")
    df2.name = os.path.basename(path_dir2)

    barInfo3 = np.loadtxt(
        path_dir3 + "/" + os.path.basename(path_dir3) + "BarInfo.csv",
        delimiter=",",
        dtype=np.int32,
    )
    df3 = pd.read_pickle(path_dir3 + "/" + os.path.basename(path_dir3) + ".pkl")
    df3.name = os.path.basename(path_dir3)

    barInfo4 = np.loadtxt(
        path_dir4 + "/" + os.path.basename(path_dir4) + "BarInfo.csv",
        delimiter=",",
        dtype=np.int32,
    )
    df4 = pd.read_pickle(path_dir4 + "/" + os.path.basename(path_dir4) + ".pkl")
    df4.name = os.path.basename(path_dir4)

    barInfo15 = np.loadtxt(
        path_dir15 + "/" + os.path.basename(path_dir15) + "BarInfo.csv",
        delimiter=",",
        dtype=np.int32,
    )
    df15 = pd.read_pickle(path_dir15 + "/" + os.path.basename(path_dir15) + ".pkl")
    df15.name = os.path.basename(path_dir15)

    # 곡 2개의 해당 마디 비교
    for i in range(14, 21):
        compare2SongUnitBy3dPlot(
            df15, df2, 7, i, barInfo1, barInfo2, zAngle=10, xAngle=45
        )

    # 곡 한개의 같은 동작하는 마디 비교
    # for i in range(10):
    # compareUnitBy3dPlot(df1, 6, 14 + i, 46 + i, barInfo1, 10, 45)
    # compareUnitBy3dPlot(df1, 4, 1 + i, 24 + i, barInfo1, 10, 45)

