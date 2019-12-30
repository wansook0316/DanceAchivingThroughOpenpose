import json
import os
import numpy as np
import time


def dataNormalize():

    # 파일 경로를 현재 위치의 input pose에서 영상 번호별로 가져온다.
    # 그리고 그걸 리스트에 담는다.
    inputData_dir = "../data/1_raw_posedata/"

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
            "{}에 있는 비디오 작업시작 : {} 중 {} 작업 진행 중".format(
                videoData_dirs[k], len(videoData_dirs), k + 1
            )
        )
        print("해당 작업을 수행했는지 확인 합니다.")
        if os.path.exists(
            "../data/2_norm_posedata/" + os.path.basename(videoData_dirs[k])
        ):
            print("해당 작업은 이미 진행 되었습니다. 다음 비디오로 넘어갑니다.")
            continue
        else:
            print("작업이 수행되지 않았습니다. 작업을 시작합니다.")

        videoOutput_dir = "../data/2_norm_posedata/" + os.path.basename(
            videoData_dirs[k]
        )

        os.mkdir(videoOutput_dir)

        filelist = []
        temp = os.listdir(videoData_dirs[k])
        for item in temp:
            if not (item.startswith(".") or item.startswith("Icon")):
                filelist.append(item)

        filelist.sort()
        poseFrameList = []

        # 전체적인 파일 경로를 만들어주는 작업이다.
        # .DS_store 제거
        for j in range(len(filelist)):
            if not (filelist[j].startswith(".") or filelist[j].startswith("Icon")):
                poseFrameList.append(videoData_dirs[k] + "/" + filelist[j])

        # print(poseFrameList)
        # print("{}의 파일 리스트를 읽었습니다.".format(videoData_dirs[k]))

        # 나중에 파일 쓰기 할 때, 몇번째 파일인지 써줄 변수를 하나 만든다.
        numbering = -1

        # filelist에 들어가 있는 json 파일 하나마다 전처리 작업을 해준다.
        setStartFrame = 0
        # if os.path.basename(videoData_dirs[k]) == "violeta5":
        #     setStartFrame = 0
        # elif os.path.basename(videoData_dirs[k]) == "girlfriend":
        #     # setStartFrame = 132
        #     # setStartFrame = 107
        #     setStartFrame = 0
        # print(setStartFrame)
        # time.sleep(10)

        for i in range(setStartFrame, len(filelist)):
            numbering += 1

            # print(poseFrameList[i])
            with open(poseFrameList[i], "r") as fr:
                try:
                    json_data = json.load(fr)
                except ValueError:
                    with open(poseFrameList[500], "r") as tempfr:
                        json_data = json.load(tempfr)

                    with open(
                        videoOutput_dir + "/{0:08d}.json".format(numbering),
                        "w",
                        encoding="utf-8",
                    ) as make_file:
                        json.dump(json_data, make_file, indent="\t")
                        json_data["people"][0]["pose_keypoints_2d"] = np.zeros(
                            len(json_data["people"][0]["pose_keypoints_2d"])
                        ).tolist()
                    continue

            # json 파일의 구조를 제대로 보기 위한 코드이다.
            # print(json.dumps(json_data, indent="\t"))

            # 만약 사람이 detection이 되지 않은 json 파일이 들어왔을 경우에, zero matrix로 파일을 쓴다.
            # 사람이 detection이 안되었을 경우에는 파일을 그대로 내보냄
            if json_data["people"] == []:
                print("{}번째 프레임 : detection에 실패했습니다. 다음 프레임으로 넘어갑니다.".format(numbering))
                # print(len(json_data["people"][0]["pose_keypoints_2d"]))
                # json_data["people"][0]["pose_keypoints_2d"] = np.zeros(
                #     len(json_data["people"][0]["pose_keypoints_2d"])
                # ).tolist()
                with open(poseFrameList[500], "r") as tempfr:
                    json_data = json.load(tempfr)

                with open(
                    videoOutput_dir + "/{0:08d}.json".format(numbering),
                    "w",
                    encoding="utf-8",
                ) as make_file:
                    json.dump(json_data, make_file, indent="\t")
                    json_data["people"][0]["pose_keypoints_2d"] = np.zeros(
                        len(json_data["people"][0]["pose_keypoints_2d"])
                    ).tolist()

                continue

            # 해당 사람의 정보를 읽어 들임, 그리고 reshape함
            poseData = np.array(json_data["people"][0]["pose_keypoints_2d"])
            # leftHandData = np.array(json_data["people"][0]["hand_left_keypoints_2d"])
            # rightHandData = np.array(json_data["people"][0]["hand_right_keypoints_2d"])
            poseData = np.reshape(poseData, (-1, 3))
            # leftHandData = np.reshape(leftHandData, (-1, 3))
            # rightHandData = np.reshape(rightHandData, (-1, 3))

            # 만약 파일을 불러오고, pose keypoint의 MID_HIP이 0인 경우 파일을 쓰지 않고 종료한다.
            if not poseData[8].any():
                print(
                    "{}번째 프레임 : MID_HIP 좌표값을 얻지 못했습니다. 다음 프레임으로 넘어갑니다.".format(
                        numbering
                    )
                )

                json_data["people"][0]["pose_keypoints_2d"] = np.zeros(
                    len(json_data["people"][0]["pose_keypoints_2d"])
                ).tolist()

                with open(
                    videoOutput_dir + "/{0:08d}.json".format(numbering),
                    "w",
                    encoding="utf-8",
                ) as make_file:
                    json.dump(json_data, make_file, indent="\t")
                    print(json_data)
                    time.sleep(1)

                continue

            # 1980 * 1080 영상으로 가정했을 때 중앙값을 설정한다.
            MID_HIP = poseData[8]
            CENTER_PIXEL = np.array([990, 540])

            # 다른 요소가 비어있을 경우 0으로 그대로 해당 열을 둔다.
            # 초기화 작업이다. 각종 예외처리를 한 후에, zero matrix에서부터 시작한다.
            poseData_norm = np.zeros(np.shape(poseData))
            # leftHandData_norm = np.zeros(np.shape(leftHandData))
            # rightHandData_norm = np.zeros(np.shape(rightHandData))

            # 추정을 잘 못하여, 너무 다른 곳에 좌표를 찍은 녀석이 있다.
            # 평행이동 후에, 값이 음수로 바뀌게 되니 이부분을 제거하고 0으로 초기화해두자.
            for i in range(np.shape(poseData)[0]):
                if not poseData[i].any():
                    poseData_norm[i] = poseData[i]
                    continue
                for j in range(np.shape(poseData)[1] - 1):
                    poseData_norm[i][j] = poseData[i][j] - MID_HIP[j] + CENTER_PIXEL[j]
                    poseData_norm[i][2] = poseData[i][2]
                if (
                    poseData_norm[i][0] < 0
                    or poseData_norm[i][1] < 0
                    or poseData_norm[i][2] < 0
                ):
                    print("{}번째 프레임 : pose에서 오류 데이터 발견".format(numbering))
                    print(poseData_norm[i])
                    poseData_norm[i][0] = 0
                    poseData_norm[i][1] = 0
                    poseData_norm[i][2] = 0
                    print(poseData_norm[i])
                    time.sleep(3)

            # for i in range(np.shape(leftHandData)[0]):
            #     if not leftHandData[i].any():
            #         leftHandData_norm[i] = leftHandData[i]
            #         continue
            #     for j in range(np.shape(leftHandData)[1] - 1):
            #         leftHandData_norm[i][j] = (
            #             leftHandData[i][j] - MID_HIP[j] + CENTER_PIXEL[j]
            #         )
            #     leftHandData_norm[i][2] = leftHandData[i][2]
            #     if (
            #         leftHandData_norm[i][0] < 0
            #         or leftHandData_norm[i][1] < 0
            #         or leftHandData_norm[i][2] < 0
            #     ):
            #         print("{}번쨰 프레임 : left에서 오류 데이터 발견".format(numbering))
            #         print(leftHandData_norm[i])
            #         leftHandData_norm[i][0] = 0
            #         leftHandData_norm[i][1] = 0
            #         leftHandData_norm[i][2] = 0
            #         print(leftHandData_norm[i])

            # for i in range(np.shape(rightHandData)[0]):
            #     if not rightHandData[i].any():
            #         rightHandData_norm[i] = rightHandData[i]
            #         continue
            #     for j in range(np.shape(rightHandData)[1] - 1):
            #         rightHandData_norm[i][j] = (
            #             rightHandData[i][j] - MID_HIP[j] + CENTER_PIXEL[j]
            #         )
            #     rightHandData_norm[i][2] = rightHandData[i][2]
            #     if (
            #         rightHandData_norm[i][0] < 0
            #         or rightHandData_norm[i][1] < 0
            #         or rightHandData_norm[i][2] < 0
            #     ):
            #         print("{}번째 프레임 : right에서 오류 데이터 발견".format(numbering))
            #         print(rightHandData_norm[i])
            #         rightHandData_norm[i][0] = 0
            #         rightHandData_norm[i][1] = 0
            #         rightHandData_norm[i][2] = 0
            #         print(rightHandData_norm[i])

            # 각 프레임의 평균과 분산을 구하는 과정이다.
            # 이 떄,  오류 데이터 행이 있으므로 그 행의 index를 저장해두자.

            delete_index_pose = []

            # delete_index_right = []
            # delete_index_left = []

            for row in range(np.shape(poseData)[0]):
                if np.all(poseData[row] == [0.0, 0.0, 0.0]):
                    delete_index_pose.append(row)
            # print(delete_index_pose)
            # for row in range(np.shape(rightHandData)[0]):
            #     if np.all(rightHandData[row] == [0.0, 0.0, 0.0]):
            #         delete_index_right.append(i)
            # for row in range(np.shape(leftHandData)[0]):
            #     if np.all(leftHandData[row] == [0.0, 0.0, 0.0]):
            #         delete_index_left.append(i)

            # 그 index를 기반으로 행을 지워주자.
            # poseData = np.delete(poseData, (delete_index_pose), axis=0)
            # rightHandData = np.delete(rightHandData, (delete_index_right), axis=0)
            # leftHandData = np.delete(leftHandData, (delete_index_left), axis=0)

            # 그리고 그 3개의 행렬을 하나로 묶어주자.
            # 이렇게 해주는 이유는 , 평균 같은 경우 [0, 0, 0]이 전체 프레임 노드의 평균을 구하는데 영향을 주지 않지만,
            # 분산같은 경우 분모 숫자가 달라지게 되어 전체 분포를 대변하기 어렵다.
            # 또한 정규화를 아래에서 진행할 경우에 명확하지 않다. 즉, 오류데이터를 분포에서 제거한 뒤 정규화를 하고자 한 것이 목표이다.
            # mean_vectorX = np.append(
            #     np.append(poseData[:, 0], rightHandData[:, 0]), leftHandData[:, 0]
            # )
            # mean_vectorY = np.append(
            #     np.append(poseData[:, 1], rightHandData[:, 1]), leftHandData[:, 1]
            # )

            mean_vectorX = poseData[:, 0]
            mean_vectorY = poseData[:, 1]

            mean_x = np.mean(mean_vectorX)
            mean_y = np.mean(mean_vectorY)
            std_x = np.std(mean_vectorX)
            std_y = np.std(mean_vectorY)

            # 평행이동이 된 변수에, 정규화를 수행한다.
            # 이 때, 전체적으로 정규화 식을 적용한 뒤에, 아까 지우려고 했던 [0,0,0] 오류 데이터의 인덱스를 재 적용한다.

            # poseData_norm[:, 0] = (poseData_norm[:, 0] - mean_x) / std_x
            # poseData_norm[:, 1] = (poseData_norm[:, 1] - mean_y) / (std_y / 2)

            poseData_norm[:, 0] = (poseData_norm[:, 0] - mean_x) / (std_x)
            poseData_norm[:, 1] = (poseData_norm[:, 1] - mean_y) / (std_y / 2)

            poseData_norm[:, 0] = poseData_norm[:, 0] - poseData_norm[8, 0]
            poseData_norm[:, 1] = poseData_norm[:, 1] - poseData_norm[8, 1]

            poseData_norm[delete_index_pose, :] = [0.0, 0.0, 0.0]

            # rightHandData_norm[:, 0] = (rightHandData_norm[:, 0] - mean_x) / std_x
            # rightHandData_norm[:, 1] = (rightHandData_norm[:, 1] - mean_y) / std_y
            # rightHandData_norm[delete_index_right, :] = [0.0, 0.0, 0.0]

            # leftHandData_norm[:, 0] = (leftHandData_norm[:, 0] - mean_x) / std_x
            # leftHandData_norm[:, 1] = (leftHandData_norm[:, 1] - mean_y) / std_y
            # leftHandData_norm[delete_index_left, :] = [0.0, 0.0, 0.0]

            # 다시 json으로 넣기 위해 1차원 배열로 만든다.
            poseData_norm = poseData_norm.flatten()
            # leftHandData_norm = leftHandData_norm.flatten()
            # rightHandData_norm = rightHandData_norm.flatten()

            json_data["people"][0]["pose_keypoints_2d"] = poseData_norm.tolist()
            # json_data["people"][0][
            #     "hand_left_keypoints_2d"
            # ] = leftHandData_norm.tolist()
            # json_data["people"][0][
            #     "hand_right_keypoints_2d"
            # ] = rightHandData_norm.tolist()

            if numbering % 500 == 0:
                print("{0} 번째 파일을 쓰는 중입니다.".format(numbering))

            with open(
                videoOutput_dir + "/{0:08d}.json".format(numbering),
                "w",
                encoding="utf-8",
            ) as make_file:
                json.dump(json_data, make_file, indent="\t")

        videoOutput_dir

        outputVideoLists = []
        temp = os.listdir(videoOutput_dir)
        for item in temp:
            if not (item.startswith(".") or item.startswith("Icon")):
                outputVideoLists.append(item)

        print()
        print("========== 파일 쓰기를 완료했습니다.=========", end="\n\n")
        print(
            "{} 비디오의 input 파일 갯수 {}개, output 파일 갯수 {}개 입니다.".format(
                os.path.basename(videoData_dirs[k]),
                len(filelist),
                len(outputVideoLists),
            ),
            end="\n\n",
        )
        time.sleep(10)
        # print("========== 파일 쓰기를 완료했습니다.=========", end = "\n\n")

