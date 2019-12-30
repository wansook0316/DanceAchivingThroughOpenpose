from data_normalization import *
from makeDataframe import *
from BaseUnitDf import *
from DataAugmentation import *
from mergeAllDf import *
import pandas as pd

dataNormalize()
makeDataFrame()

# "../data/4_dataAugmentation/"


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

path_dir16 = "../data/3_dataFrame/105b"

trueBarInfo_path_dir = "../data/trueBarInfo_30sm"

path_dir_lists = [
    path_dir1,
    path_dir2,
    path_dir3,
    path_dir4,
    path_dir5,
    path_dir6,
    path_dir8,
    path_dir9,
    path_dir10,
    path_dir11,
    path_dir12,
    path_dir13,
    path_dir14,
    path_dir15,
    path_dir16
]


for path_dir in path_dir_lists:
    makeBasicBarUnitDf(path_dir, trueBarInfo_path_dir)

for path_dir in path_dir_lists:
    makeAugmentaedDf_fromBasicBarUnit(path_dir)
