# import cv2
# def get_video_fps(infilename):

#     cap = cv2.VideoCapture(infilename)

#     if not cap.isOpened():
#         print("could not open :", infilename)
#         exit(0)

#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     print('length : ', length)
#     print('width : ', width)
#     print('height : ', height)
#     print('fps : ', fps)

import cv2


def get_file_fps(path):
    video = cv2.VideoCapture(path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print(
            # "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {}".format(fps)
        )
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {}".format(fps))

    video.release()
    return fps


# if __name__ == "__main__":

#     video = cv2.VideoCapture("../data/0_default_video/girlfriend.mp4")

#     # Find OpenCV version
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

#     if int(major_ver) < 3:
#         fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
#         print(
#             "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {}".format(fps)
#         )
#     else:
#         fps = video.get(cv2.CAP_PROP_FPS)
#         print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {}".format(fps))

#     video.release()

