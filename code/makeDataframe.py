from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.utils.testing import ignore_warnings
import os
import json
import numpy as np
import pandas as pd
import math as m
import BPM_checker as BC
import time
import FPS_checker as FC

# 이 부분은 사용자가 노래를 들어보고, 이 노래의 동작 단위가 어떤지 생각하고 넣어주어야 함


def makeDataFrame(quarterNoteCount=4):
    # bpm = BC.get_file_bpm("../data/0_default_video")
    # bpmFileLists = []
    # temp = os.listdir(bpmFileData_dir)
    # for item in temp:
    #     if not item.sat'("."):
    #         bpmFileLists.append(item)

    # bpmFileData_dirs = []
    # for i in range(len(bpmFileLists)):
    #     bpmFileData_dirs.append(bpmFileData_dir + bpmFileLists[i])

    bpmFileData_dir = "../data/0_default_video/"
    inputData_dir = "../data/2_norm_posedata/"
    # .Ds_store 파일 제거
    videoLists = []
    temp = os.listdir(inputData_dir)
    for item in temp:
        if not (item.startswith(".") or item.startswith("Icon")):
            videoLists.append(item)

    videoData_dirs = []

    for i in range(len(videoLists)):
        videoData_dirs.append(inputData_dir + videoLists[i])

    for k in range(len(videoData_dirs)):
        print(
            "{}에 있는 비디오 dataframe화 작업시작 : {} 중 {} 작업 진행 중".format(
                videoData_dirs[k], len(videoData_dirs), k + 1
            )
        )
        print("해당 작업을 수행했는지 확인 합니다.")
        if os.path.exists("../data/3_dataFrame/" + os.path.basename(videoData_dirs[k])):
            print("해당 작업은 이미 진행 되었습니다. 다음 비디오로 넘어갑니다.")
            continue
        else:
            print("작업이 수행되지 않았습니다. 작업을 시작합니다.")

        videoOutput_dir = "../data/3_dataFrame/" + os.path.basename(videoData_dirs[k])
        os.mkdir(videoOutput_dir)

        filelist = os.listdir(videoData_dirs[k])
        filelist.sort()

        bpm = BC.get_file_bpm(
            bpmFileData_dir + os.path.basename(videoData_dirs[k]) + ".mp4"
        )
        # fps = 23.4
        fps = FC.get_file_fps(
            bpmFileData_dir + os.path.basename(videoData_dirs[k]) + ".mp4"
        )
        secondPerQuarterNote = 60 / bpm
        secondPer1Unit = secondPerQuarterNote * quarterNoteCount
        framePer1Unit = secondPer1Unit * fps
        framesecondPer1Unit = round(secondPer1Unit / framePer1Unit, 3)
        totalBarLength = int(len(filelist) / framePer1Unit)

        print(
            "=========== {} 동영상 Infomation ==========".format(
                os.path.basename(videoData_dirs[k])
            )
        )

        # 5081개........
        print("파일 갯수 : {}".format(len(filelist)))
        print("bpm : {}".format(bpm))
        print("fps : {}".format(fps))
        print("4분 음표당 걸리는 시간 : {}".format(secondPerQuarterNote))
        print("한 마디당 걸리는 시간 : {}".format(secondPer1Unit))
        print("한 마디에 종속되는 프레임 갯수 : {}".format(framePer1Unit))
        print("1마디에 속해있는 한 프레임에 주어야 하는 시간 차  : {}".format(framesecondPer1Unit))
        print("총 마디 갯수 : {}".format(totalBarLength))
        time.sleep(3)

        poseFrameList = []
        frameListPer1Unit = np.array([])
        for j in range(len(filelist)):
            poseFrameList.append(videoData_dirs[k] + "/" + filelist[j])

        totalData = np.array([])
        for i in range(totalBarLength):
            newFramePer1Unit = int(framePer1Unit * (i + 1) - frameListPer1Unit.sum())
            frameListPer1Unit = np.append(frameListPer1Unit, newFramePer1Unit)

            for j in range(newFramePer1Unit):
                frameTimeDiff = framesecondPer1Unit * j

                # print(poseFrameList[int(frameListPer1Unit[:-1].sum()) + j])
                with open(
                    poseFrameList[int(frameListPer1Unit[:-1].sum()) + j], "r"
                ) as fr:
                    json_data = json.load(fr)

                # 파일을 읽었을 때, detection 하지 못한 경우 예외처리
                if json_data["people"] == []:
                    poseData = np.zeros(50)
                    temp = np.append(poseData, frameTimeDiff)
                    totalData = np.append(totalData, temp.reshape((1, -1)))
                    continue

                poseData = np.array(json_data["people"][0]["pose_keypoints_2d"])
                # leftHandData = np.array(
                #     json_data["people"][0]["hand_left_keypoints_2d"]
                # )
                # rightHandData = np.array(
                #     json_data["people"][0]["hand_right_keypoints_2d"]
                # )
                poseData = np.reshape(poseData, (-1, 3))
                # leftHandData = np.reshape(leftHandData, (-1, 3))
                # rightHandData = np.reshape(rightHandData, (-1, 3))

                poseData = np.delete(poseData, 2, 1)
                # leftHandData = np.delete(leftHandData, 2, 1)
                # rightHandData = np.delete(rightHandData, 2, 1)

                poseData = np.reshape(poseData, (1, -1))
                # leftHandData = np.reshape(leftHandData, (1, -1))
                # rightHandData = np.reshape(rightHandData, (1, -1))
                poseData = poseData.flatten()
                # leftHandData = leftHandData.flatten()
                # rightHandData = rightHandData.flatten()

                poseKeypoint_len = int(poseData.shape[0] / 2)
                # leftHandKeypoint_len = int(leftHandData.shape[0] / 2)
                # rightHandKeypoint_len = int(rightHandData.shape[0] / 2)

                # temp = np.append(
                #     np.append(np.append(poseData, rightHandData), leftHandData),
                #     frameTimeDiff,
                # )
                temp = np.append(poseData, frameTimeDiff)
                #     print(temp.shape)

                totalData = np.append(totalData, temp.reshape((1, -1)))

        totalData = totalData.reshape((-1, temp.shape[0]))
        print(totalData.shape)
        print(totalData)

        print("{}파일의 마디 정보를 저장합니다.".format(os.path.basename(videoData_dirs[k])))
        np.savetxt(
            videoOutput_dir + "/" + os.path.basename(videoData_dirs[k]) + "BarInfo.csv",
            frameListPer1Unit,
            delimiter=",",
        )

        pose_col = np.array(
            [["pose:x_" + str(i), "pose:y_" + str(i)] for i in range(poseKeypoint_len)]
        )
        # rightHand_col = np.array(
        #     [
        #         ["rightHand:x_" + str(i + 1), "rightHand:y_" + str(i + 1)]
        #         for i in range(rightHandKeypoint_len)
        #     ]
        # )
        # leftHand_col = np.array(
        #     [
        #         ["leftHand:x_" + str(i + 1), "leftHand:y_" + str(i + 1)]
        #         for i in range(leftHandKeypoint_len)
        #     ]
        # )
        pose_col = pose_col.flatten()
        # rightHand_col = rightHand_col.flatten()
        # leftHand_col = leftHand_col.flatten()

        # print(pose_col.shape)
        # print(leftHand_col.shape)
        # print(rightHand_col.shape)

        total_col = np.append(
            # np.append(np.append(pose_col, rightHand_col), leftHand_col),
            # "frameTimeDiffPer1Unit",
            pose_col,
            "frameTimeDiffPer1Unit",
        )
        # print(total_col)

        df = pd.DataFrame(data=totalData, columns=total_col)

        df.to_pickle(
            videoOutput_dir + "/" + os.path.basename(videoData_dirs[k]) + ".pkl"
        )
        print()
        print("=========== 데이터화 작업 완료 ===========")
        print("{} 동영상을 데이터화 했습니다.".format(os.path.basename(videoData_dirs[k])))


# inputData_dir = "../data/2_raw_posedata/"

#     # .Ds_store 파일 제거
#     videoLists = []
#     temp = os.listdir(inputData_dir)
#     for item in temp:
#         if not item.startswith("."):
#             videoLists.append(item)

# working_dir = "../data/2_norm_posedata" +
# filelist = os.listdir(working_dir)
# filelist.sort()
# filelist = filelist[180:]
# # print(filelist)
# # print(len(filelist) / 57)

# poseFrameList = []

# for i in range(len(filelist)):
#     poseFrameList.append(working_dir + filelist[i])

# print("해당 폴더의 파일 리스트를 읽었습니다.")


# bpm = 125
# fps = 30
# secondPer4 = 60 / 125
# secondPer1Bar = secondPer4 * 4
# framePer1Bar = int(secondPer1Bar * fps)
# framesecondPer1Bar = round(secondPer1Bar / framePer1Bar, 3)

# totalBarLength =  int(len(filelist)/framePer1Bar)

# print("bpm : {}".format(bpm))
# print("4분 음표당 걸리는 시간 : {}".format(secondPer4))
# print("한 마디당 걸리는 시간 : {}".format(secondPer1Bar))
# print("한 마디에 종속되는 프레임 갯수 : {}".format(framePer1Bar))
# print("1마디에 속해있는 한 프레임에 주어야 하는 시간 차  : {}".format(framesecondPer1Bar))

# print("총 마디 갯수 : {}".format(totalBarLength))

