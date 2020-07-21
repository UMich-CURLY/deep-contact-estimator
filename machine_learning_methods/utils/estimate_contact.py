from machine_learning_methods.utils.Jp_VectorNav_to_RightToeBottom import*
from machine_learning_methods.utils.Jp_VectorNav_to_LeftToeBottom import*


def estimateContacts(encoders, contact_threshold):
    J_pL = Jp_VectorNav_to_LeftToeBottom(encoders)
    J_pR = Jp_VectorNav_to_RightToeBottom(encoders)

    qs_knee_L = encoders[4]
    qs_tarsus_L = encoders[3] + encoders[4] + encoders[5] - 0.2269

    qs_knee_R = encoders[11]
    qs_tarsus_R = encoders[10] + encoders[11] + encoders[12] - 0.2269

    K = np.array([[1500], [1250]])
    spring_deflection_left = np.array([[qs_knee_L + qs_tarsus_L], [qs_tarsus_L]])
    spring_deflection_right = np.array([[qs_knee_R + qs_tarsus_R], [qs_tarsus_R]])
    tau_L = -K*spring_deflection_left
    tau_R = -K*spring_deflection_right

    JL = np.array([[J_pL[0, 4], J_pL[0, 5]],
                   [J_pL[2, 4], J_pL[2, 5]]])

    JR = np.array([[J_pR[0, 11], J_pR[0, 12]],
                   [J_pR[2, 11], J_pR[2, 12]]])

    GRF_L = np.dot(-np.linalg.inv(JL), tau_L)
    GRF_R = np.dot(-np.linalg.inv(JR), tau_R)

    contact_left = GRF_L[1, 0] > contact_threshold
    contact_right = GRF_R[1, 0] > contact_threshold

    return contact_left, contact_right