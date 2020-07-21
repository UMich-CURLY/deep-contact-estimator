import numpy as np

def Jp_VectorNav_to_LeftToeBottom(var1):
    t128 = np.sin(var1[0])
    t286 = np.cos(var1[1])
    t750 = np.sin(var1[1])
    t956 = np.cos(var1[2])
    t1009 = -1. * t956
    t1015 = 1. + t1009
    t1151 = np.sin(var1[2])
    t1428 = np.cos(var1[0])
    t1850 = np.cos(var1[3])
    t1966 = -1. * t1850
    t2067 = 1. + t1966
    t2307 = np.sin(var1[3])
    t2538 = -1. * t956 * t128 * t750
    t2624 = -1. * t1428 * t1151
    t2640 = t2538 + t2624
    t2858 = -1. * t1428 * t956
    t2905 = t128 * t750 * t1151
    t2918 = t2858 + t2905
    t2962 = np.cos(var1[4])
    t2978 = -1. * t2962
    t2981 = 1. + t2978
    t3021 = np.sin(var1[4])
    t3061 = -1. * t2307 * t2640
    t3094 = t1850 * t2918
    t3121 = t3061 + t3094
    t3343 = t1850 * t2640
    t3429 = t2307 * t2918
    t3451 = t3343 + t3429
    t3485 = np.cos(var1[5])
    t3499 = -1. * t3485
    t3500 = 1. + t3499
    t3557 = np.sin(var1[5])
    t3737 = t3021 * t3121
    t3820 = t2962 * t3451
    t3855 = t3737 + t3820
    t3948 = t2962 * t3121
    t3971 = -1. * t3021 * t3451
    t3995 = t3948 + t3971
    t4067 = np.cos(var1[6])
    t4111 = -1. * t4067
    t4135 = 1. + t4111
    t4179 = np.sin(var1[6])
    t4203 = -1. * t3557 * t3855
    t4207 = t3485 * t3995
    t4218 = t4203 + t4207
    t4297 = t3485 * t3855
    t4301 = t3557 * t3995
    t4302 = t4297 + t4301
    t587 = -1. * t286
    t604 = 1. + t587
    t677 = 0.135 * t604
    t802 = 0.049 * t750
    t860 = 0. + t677 + t802
    t1131 = -0.049 * t1015
    t1248 = -0.09 * t1151
    t1337 = 0. + t1131 + t1248
    t1481 = -0.09 * t1015
    t1525 = 0.049 * t1151
    t1553 = 0. + t1481 + t1525
    t2234 = -0.049 * t2067
    t2331 = -0.21 * t2307
    t2365 = 0. + t2234 + t2331
    t2686 = -0.21 * t2067
    t2710 = 0.049 * t2307
    t2714 = 0. + t2686 + t2710
    t3017 = -0.2707 * t2981
    t3046 = 0.0016 * t3021
    t3050 = 0. + t3017 + t3046
    t4623 = t1428 * t956 * t750
    t4627 = -1. * t128 * t1151
    t4663 = t4623 + t4627
    t4715 = -1. * t956 * t128
    t4736 = -1. * t1428 * t750 * t1151
    t4792 = t4715 + t4736
    t3157 = -0.0016 * t2981
    t3234 = -0.2707 * t3021
    t3338 = 0. + t3157 + t3234
    t3504 = 0.0184 * t3500
    t3618 = -0.7055 * t3557
    t3642 = 0. + t3504 + t3618
    t4832 = -1. * t2307 * t4663
    t4840 = t1850 * t4792
    t4848 = t4832 + t4840
    t4863 = t1850 * t4663
    t4865 = t2307 * t4792
    t4906 = t4863 + t4865
    t3872 = -0.7055 * t3500
    t3876 = -0.0184 * t3557
    t3887 = 0. + t3872 + t3876
    t4139 = -1.1135 * t4135
    t4187 = 0.0216 * t4179
    t4198 = 0. + t4139 + t4187
    t4941 = t3021 * t4848
    t4956 = t2962 * t4906
    t4958 = t4941 + t4956
    t4966 = t2962 * t4848
    t4978 = -1. * t3021 * t4906
    t4994 = t4966 + t4978
    t4253 = -0.0216 * t4135
    t4263 = -1.1135 * t4179
    t4293 = 0. + t4253 + t4263
    t5013 = -1. * t3557 * t4958
    t5015 = t3485 * t4994
    t5018 = t5013 + t5015
    t5032 = t3485 * t4958
    t5051 = t3557 * t4994
    t5063 = t5032 + t5051
    t5229 = t956 * t2307 * t750
    t5241 = t1850 * t750 * t1151
    t5248 = t5229 + t5241
    t5252 = -1. * t1850 * t956 * t750
    t5256 = t2307 * t750 * t1151
    t5272 = t5252 + t5256
    t5283 = t3021 * t5248
    t5284 = t2962 * t5272
    t5286 = t5283 + t5284
    t5320 = t2962 * t5248
    t5327 = -1. * t3021 * t5272
    t5331 = t5320 + t5327
    t5345 = -1. * t3557 * t5286
    t5352 = t3485 * t5331
    t5356 = t5345 + t5352
    t5360 = t3485 * t5286
    t5361 = t3557 * t5331
    t5363 = t5360 + t5361
    t5497 = -1. * t1428 * t286 * t956 * t2307
    t5508 = -1. * t1850 * t1428 * t286 * t1151
    t5518 = t5497 + t5508
    t5523 = t1850 * t1428 * t286 * t956
    t5525 = -1. * t1428 * t286 * t2307 * t1151
    t5527 = t5523 + t5525
    t5533 = t3021 * t5518
    t5547 = t2962 * t5527
    t5552 = t5533 + t5547
    t5561 = t2962 * t5518
    t5572 = -1. * t3021 * t5527
    t5580 = t5561 + t5572
    t5593 = -1. * t3557 * t5552
    t5597 = t3485 * t5580
    t5600 = t5593 + t5597
    t5613 = t3485 * t5552
    t5614 = t3557 * t5580
    t5622 = t5613 + t5614
    t5433 = 0.049 * t286
    t5451 = 0.135 * t750
    t5466 = t5433 + t5451
    t5737 = -1. * t286 * t956 * t2307 * t128
    t5741 = -1. * t1850 * t286 * t128 * t1151
    t5744 = t5737 + t5741
    t5747 = t1850 * t286 * t956 * t128
    t5757 = -1. * t286 * t2307 * t128 * t1151
    t5767 = t5747 + t5757
    t5785 = t3021 * t5744
    t5786 = t2962 * t5767
    t5790 = t5785 + t5786
    t5796 = t2962 * t5744
    t5797 = -1. * t3021 * t5767
    t5804 = t5796 + t5797
    t5814 = -1. * t3557 * t5790
    t5827 = t3485 * t5804
    t5832 = t5814 + t5827
    t5845 = t3485 * t5790
    t5846 = t3557 * t5804
    t5851 = t5845 + t5846
    t5923 = -1. * t286 * t956 * t2307
    t5932 = -1. * t1850 * t286 * t1151
    t5938 = t5923 + t5932
    t5950 = -1. * t1850 * t286 * t956
    t5958 = t286 * t2307 * t1151
    t5966 = t5950 + t5958
    t5991 = -1. * t3021 * t5938
    t6007 = t2962 * t5966
    t6010 = t5991 + t6007
    t6028 = t2962 * t5938
    t6034 = t3021 * t5966
    t6045 = t6028 + t6034
    t6056 = t3557 * t6010
    t6061 = t3485 * t6045
    t6062 = t6056 + t6061
    t6079 = t3485 * t6010
    t6082 = -1. * t3557 * t6045
    t6084 = t6079 + t6082
    t5872 = -0.09 * t956
    t5876 = -0.049 * t1151
    t5893 = t5872 + t5876
    t6138 = -1. * t1428 * t956 * t750
    t6159 = t128 * t1151
    t6160 = t6138 + t6159
    t6171 = t2307 * t6160
    t6172 = t6171 + t4840
    t6181 = t1850 * t6160
    t6183 = -1. * t2307 * t4792
    t6184 = t6181 + t6183
    t6188 = -1. * t3021 * t6172
    t6192 = t2962 * t6184
    t6195 = t6188 + t6192
    t6198 = t2962 * t6172
    t6199 = t3021 * t6184
    t6211 = t6198 + t6199
    t6216 = t3557 * t6195
    t6219 = t3485 * t6211
    t6225 = t6216 + t6219
    t6233 = t3485 * t6195
    t6235 = -1. * t3557 * t6211
    t6249 = t6233 + t6235
    t6122 = 0.049 * t956
    t6130 = t6122 + t1248
    t6285 = t1428 * t956
    t6287 = -1. * t128 * t750 * t1151
    t6288 = t6285 + t6287
    t6291 = t2307 * t2640
    t6292 = t1850 * t6288
    t6294 = t6291 + t6292
    t6297 = -1. * t2307 * t6288
    t6298 = t3343 + t6297
    t6308 = -1. * t3021 * t6294
    t6309 = t2962 * t6298
    t6316 = t6308 + t6309
    t6323 = t2962 * t6294
    t6325 = t3021 * t6298
    t6329 = t6323 + t6325
    t6335 = t3557 * t6316
    t6336 = t3485 * t6329
    t6351 = t6335 + t6336
    t6355 = t3485 * t6316
    t6359 = -1. * t3557 * t6329
    t6362 = t6355 + t6359
    t5949 = t3338 * t5938
    t5974 = t3050 * t5966
    t6016 = t3887 * t6010
    t6051 = t3642 * t6045
    t6070 = t4293 * t6062
    t6086 = t4198 * t6084
    t6088 = -1. * t4179 * t6062
    t6096 = t4067 * t6084
    t6100 = t6088 + t6096
    t6101 = -1.1312 * t6100
    t6105 = t4067 * t6062
    t6107 = t4179 * t6084
    t6108 = t6105 + t6107
    t6111 = 0.0306 * t6108
    t6403 = -0.21 * t1850
    t6405 = -0.049 * t2307
    t6407 = t6403 + t6405
    t6414 = 0.049 * t1850
    t6415 = t6414 + t2331
    t6433 = -1. * t1850 * t4663
    t6434 = t6433 + t6183
    t6439 = -1. * t3021 * t4848
    t6440 = t2962 * t6434
    t6445 = t6439 + t6440
    t6448 = t3021 * t6434
    t6449 = t4966 + t6448
    t6463 = t3557 * t6445
    t6466 = t3485 * t6449
    t6496 = t6463 + t6466
    t6506 = t3485 * t6445
    t6510 = -1. * t3557 * t6449
    t6511 = t6506 + t6510
    t6567 = t956 * t128 * t750
    t6572 = t1428 * t1151
    t6574 = t6567 + t6572
    t6591 = -1. * t2307 * t6574
    t6599 = t6591 + t6292
    t6610 = -1. * t1850 * t6574
    t6613 = t6610 + t6297
    t6616 = -1. * t3021 * t6599
    t6617 = t2962 * t6613
    t6620 = t6616 + t6617
    t6627 = t2962 * t6599
    t6628 = t3021 * t6613
    t6632 = t6627 + t6628
    t6638 = t3557 * t6620
    t6642 = t3485 * t6632
    t6644 = t6638 + t6642
    t6647 = t3485 * t6620
    t6648 = -1. * t3557 * t6632
    t6654 = t6647 + t6648
    t6713 = t1850 * t286 * t956
    t6727 = -1. * t286 * t2307 * t1151
    t6731 = t6713 + t6727
    t6744 = -1. * t2962 * t6731
    t6748 = t5991 + t6744
    t6759 = -1. * t3021 * t6731
    t6764 = t6028 + t6759
    t6772 = t3557 * t6748
    t6773 = t3485 * t6764
    t6775 = t6772 + t6773
    t6778 = t3485 * t6748
    t6781 = -1. * t3557 * t6764
    t6782 = t6778 + t6781
    t6694 = 0.0016 * t2962
    t6695 = t6694 + t3234
    t6702 = -0.2707 * t2962
    t6707 = -0.0016 * t3021
    t6710 = t6702 + t6707
    t6826 = -1. * t2962 * t4906
    t6830 = t6439 + t6826
    t6845 = t3557 * t6830
    t6848 = t6845 + t5015
    t6852 = t3485 * t6830
    t6853 = -1. * t3557 * t4994
    t6854 = t6852 + t6853
    t6885 = t1850 * t6574
    t6891 = t2307 * t6288
    t6898 = t6885 + t6891
    t6901 = -1. * t2962 * t6898
    t6926 = t6616 + t6901
    t6934 = -1. * t3021 * t6898
    t6944 = t6627 + t6934
    t6957 = t3557 * t6926
    t6962 = t3485 * t6944
    t6963 = t6957 + t6962
    t6965 = t3485 * t6926
    t6968 = -1. * t3557 * t6944
    t6969 = t6965 + t6968
    t7023 = t3021 * t5938
    t7025 = t2962 * t6731
    t7032 = t7023 + t7025
    t7044 = -1. * t3557 * t7032
    t7046 = t7044 + t6773
    t7051 = -1. * t3485 * t7032
    t7059 = t7051 + t6781
    t7004 = -0.7055 * t3485
    t7005 = 0.0184 * t3557
    t7014 = t7004 + t7005
    t7040 = -0.0184 * t3485
    t7041 = t7040 + t3618
    t7113 = -1. * t3485 * t4958
    t7118 = t7113 + t6853
    t5154 = t4067 * t5018
    t7139 = t3021 * t6599
    t7140 = t2962 * t6898
    t7145 = t7139 + t7140
    t7153 = -1. * t3557 * t7145
    t7154 = t7153 + t6962
    t7157 = -1. * t3485 * t7145
    t7159 = t7157 + t6968
    t7074 = -1. * t4179 * t7046
    t7213 = t3485 * t7032
    t7216 = t3557 * t6764
    t7217 = t7213 + t7216
    t7083 = t4067 * t7046
    t7192 = 0.0216 * t4067
    t7195 = t7192 + t4263
    t7201 = -1.1135 * t4067
    t7203 = -0.0216 * t4179
    t7204 = t7201 + t7203
    t7121 = -1. * t4179 * t5018
    t5156 = -1. * t4179 * t5063
    t5166 = t5154 + t5156
    t7165 = -1. * t4179 * t7154
    t7262 = t3485 * t7145
    t7263 = t3557 * t6944
    t7264 = t7262 + t7263
    t7174 = t4067 * t7154

    p_output1 = np.zeros((42,))
    p_output1[0] = 0
    p_output1[1] = 0.135 * t128 - 1. * t1428 * t1553 + t2365 * t2640 - 0.1305 * t128 * t286 + t2714 * t2918 + t3050 * t3121 + t3338 * t3451 + t3642 * t3855 + t3887 * t3995 + t4198 * t4218 + t4293 * t4302 + 0.0306 * (
                t4179 * t4218 + t4067 * t4302) - 1.1312 * (
                         t4067 * t4218 - 1. * t4179 * t4302) - 1. * t128 * t1337 * t750 - 1. * t128 * t860
    p_output1[2]= -0.135 * t1428 - 1. * t128 * t1553 + 0.1305 * t1428 * t286 + t2365 * t4663 + t2714 * t4792 + t3050 * t4848 + t3338 * t4906 + t3642 * t4958 + t3887 * t4994 + t4198 * t5018 + t4293 * t5063 + 0.0306 * (
                t4179 * t5018 + t4067 * t5063) - 1.1312 * t5166 + t1337 * t1428 * t750 + t1428 * t860
    p_output1[3] = 0.004500000000000004 * t286 + t3050 * t5248 + t3338 * t5272 + t3642 * t5286 + t3887 * t5331 + t4198 * t5356 + t4293 * t5363 + 0.0306 * (
                t4179 * t5356 + t4067 * t5363) - 1.1312 * (
                         t4067 * t5356 - 1. * t4179 * t5363) - 0.049 * t750 - 1. * t1337 * t750 + t1151 * t2714 * t750 - 1. * t2365 * t750 * t956;
    p_output1[4] = t1337 * t1428 * t286 - 1. * t1151 * t1428 * t2714 * t286 + t1428 * t5466 + t3050 * t5518 + t3338 * t5527 + t3642 * t5552 + t3887 * t5580 + t4198 * t5600 + t4293 * t5622 + 0.0306 * (
                t4179 * t5600 + t4067 * t5622) - 1.1312 * (
                         t4067 * t5600 - 1. * t4179 * t5622) - 0.1305 * t1428 * t750 + t1428 * t2365 * t286 * t956
    p_output1[5] = t128 * t1337 * t286 - 1. * t1151 * t128 * t2714 * t286 + t128 * t5466 + t3050 * t5744 + t3338 * t5767 + t3642 * t5790 + t3887 * t5804 + t4198 * t5832 + t4293 * t5851 + 0.0306 * (
                t4179 * t5832 + t4067 * t5851) - 1.1312 * (
                         t4067 * t5832 - 1. * t4179 * t5851) - 0.1305 * t128 * t750 + t128 * t2365 * t286 * t956
    p_output1[6] = -1. * t1151 * t2365 * t286 + t286 * t5893 + t5949 + t5974 + t6016 + t6051 + t6070 + t6086 + t6101 + t6111 - 1. * t2714 * t286 * t956
    p_output1[7] = t2365 * t4792 - 1. * t128 * t6130 + t2714 * t6160 + t3338 * t6172 + t3050 * t6184 + t3887 * t6195 + t3642 * t6211 + t4293 * t6225 + t4198 * t6249 - 1.1312 * (
                -1. * t4179 * t6225 + t4067 * t6249) + 0.0306 * (t4067 * t6225 + t4179 * t6249) + t1428 * t5893 * t750
    p_output1[8] = t2640 * t2714 + t1428 * t6130 + t2365 * t6288 + t3338 * t6294 + t3050 * t6298 + t3887 * t6316 + t3642 * t6329 + t4293 * t6351 + t4198 * t6362 - 1.1312 * (
                -1. * t4179 * t6351 + t4067 * t6362) + 0.0306 * (t4067 * t6351 + t4179 * t6362) + t128 * t5893 * t750
    p_output1[9] = t5949 + t5974 + t6016 + t6051 + t6070 + t6086 + t6101 + t6111 - 1. * t1151 * t286 * t6415 + t286 * t6407 * t956
    p_output1[10] = t3338 * t4848 + t4663 * t6407 + t4792 * t6415 + t3050 * t6434 + t3887 * t6445 + t3642 * t6449 + t4293 * t6496 + t4198 * t6511 - 1.1312 * (
                -1. * t4179 * t6496 + t4067 * t6511) + 0.0306 * (t4067 * t6496 + t4179 * t6511)
    p_output1[11] = t6288 * t6415 + t6407 * t6574 + t3338 * t6599 + t3050 * t6613 + t3887 * t6620 + t3642 * t6632 + t4293 * t6644 + t4198 * t6654 - 1.1312 * (
                -1. * t4179 * t6644 + t4067 * t6654) + 0.0306 * (t4067 * t6644 + t4179 * t6654)
    p_output1[12] = t5938 * t6695 + t6710 * t6731 + t3887 * t6748 + t3642 * t6764 + t4293 * t6775 + t4198 * t6782 - 1.1312 * (
                -1. * t4179 * t6775 + t4067 * t6782) + 0.0306 * (t4067 * t6775 + t4179 * t6782)
    p_output1[13] = t3642 * t4994 + t4848 * t6695 + t4906 * t6710 + t3887 * t6830 + t4293 * t6848 + t4198 * t6854 - 1.1312 * (
                -1. * t4179 * t6848 + t4067 * t6854) + 0.0306 * (t4067 * t6848 + t4179 * t6854)
    p_output1[14] = t6599 * t6695 + t6710 * t6898 + t3887 * t6926 + t3642 * t6944 + t4293 * t6963 + t4198 * t6969 - 1.1312 * (
                -1. * t4179 * t6963 + t4067 * t6969) + 0.0306 * (t4067 * t6963 + t4179 * t6969)
    p_output1[15] = t7014 * t7032 + t6764 * t7041 + t4293 * t7046 + t4198 * t7059 - 1.1312 * (
                t4067 * t7059 + t7074) + 0.0306 * (t4179 * t7059 + t7083)
    p_output1[16] = t4293 * t5018 + t4958 * t7014 + t4994 * t7041 + t4198 * t7118 + 0.0306 * (
                t5154 + t4179 * t7118) - 1.1312 * (t4067 * t7118 + t7121)
    p_output1[17] = t6944 * t7041 + t7014 * t7145 + t4293 * t7154 + t4198 * t7159 - 1.1312 * (
                t4067 * t7159 + t7165) + 0.0306 * (t4179 * t7159 + t7174)
    p_output1[18] = t7046 * t7195 + t7204 * t7217 - 1.1312 * (t7074 - 1. * t4067 * t7217) + 0.0306 * (
                t7083 - 1. * t4179 * t7217)
    p_output1[19] = 0.0306 * t5166 - 1.1312 * (-1. * t4067 * t5063 + t7121) + t5018 * t7195 + t5063 * t7204
    p_output1[20] = t7154 * t7195 + t7204 * t7264 - 1.1312 * (t7165 - 1. * t4067 * t7264) + 0.0306 * (
                t7174 - 1. * t4179 * t7264)
    p_output1[21] = 0
    p_output1[22] = 0
    p_output1[23] = 0
    p_output1[24] = 0
    p_output1[25] = 0
    p_output1[26] = 0
    p_output1[27] = 0
    p_output1[28] = 0
    p_output1[29] = 0
    p_output1[30] = 0
    p_output1[31] = 0
    p_output1[32] = 0
    p_output1[33] = 0
    p_output1[34] = 0
    p_output1[35] = 0
    p_output1[36] = 0
    p_output1[37] = 0
    p_output1[38] = 0
    p_output1[39] = 0
    p_output1[40] = 0
    p_output1[41] = 0

    p_output1 = p_output1.reshape(14, 3).T

    return p_output1

