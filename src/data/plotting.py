import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import pytorch3d.transforms

def parse_line(line):
    parts = line.split()
    name = parts[0]
    time = float(parts[1])
    data = np.array([float(x) for x in parts[2:]])
    return name, time, data

directory = "/home/syr/github/EKF/invariant-ekf/src/data/output_data"

## mlp
mlp_R_true = []
mlp_R_pred = []
mlp_uncertainty = []

with open(directory+"/mlp_est_data.txt", "r") as file:
    for line in file:
        name, time, data = parse_line(line)
        if name=='True_Rotation':
            mlp_R_true.append(data)
        elif name=='Pred_Rotation':
            mlp_R_pred.append(data)
        elif name=='Uncertainty':
            mlp_uncertainty.append(data)

mlp_R_true = torch.tensor(mlp_R_true, dtype=torch.float32)
mlp_R_pred = torch.tensor(mlp_R_pred, dtype=torch.float32)
mlp_residual = pytorch3d.transforms.matrix_to_axis_angle(torch.matmul(mlp_R_pred.view(-1,3,3).transpose(1,2), mlp_R_true.view(-1,3,3))).detach().cpu().numpy()
mlp_uncertainty = np.array(mlp_uncertainty)


## gru
gru_R_true = []
gru_R_pred = []
gru_uncertainty = []

with open(directory+"/gru_est_data.txt", "r") as file:
    for line in file:
        name, time, data = parse_line(line)
        if name=='True_Rotation':
            gru_R_true.append(data)
        elif name=='Pred_Rotation':
            gru_R_pred.append(data)
        elif name=='Uncertainty':
            gru_uncertainty.append(data)

gru_R_true = torch.tensor(gru_R_true, dtype=torch.float32)
gru_R_pred = torch.tensor(gru_R_pred, dtype=torch.float32)
gru_residual = pytorch3d.transforms.matrix_to_axis_angle(torch.matmul(gru_R_pred.view(-1,3,3).transpose(1,2), gru_R_true.view(-1,3,3))).detach().cpu().numpy()
gru_uncertainty = np.array(gru_uncertainty)


## InEKF: DRCD_Learning_Based_Estimator-Raisin 의 save branch에서 Estimator_Tester 를 돌린 InEKF 결과
InEKF_R_true = []
InEKF_R_pred = []
InEKF_uncertainty = []

with open(directory+"/InEKF_est_data.txt", "r") as file:
    for line in file:
        name, time, data = parse_line(line)
        if name=='True_Rotation':
            InEKF_R_true.append(data)
        elif name=='Pred_Rotation':
            InEKF_R_pred.append(data)
        elif name=='Uncertainty':
            InEKF_uncertainty.append(data)

InEKF_R_true = torch.tensor(InEKF_R_true, dtype=torch.float32).view(-1,3,3)
InEKF_R_pred = torch.tensor(InEKF_R_pred, dtype=torch.float32).view(-1,3,3)
InEKF_R_temp = InEKF_R_pred[:-1,:,:]
InEKF_R_pred[0,:,:] = InEKF_R_true[0,:,:]
InEKF_R_pred[1:,:,:] = InEKF_R_temp.clone()
InEKF_residual = pytorch3d.transforms.matrix_to_axis_angle(torch.matmul(InEKF_R_pred.transpose(1,2),InEKF_R_true)).detach().cpu().numpy()
InEKF_uncertainty = np.array(InEKF_uncertainty)

## Contact InEKF: .../src/example/kinematics.cpp를 돌렸을 때 얻어지는 InEKF 결과
t = []
contactInEKF_R_true = []
contactInEKF_R_pred_T = []
contactInEKF_uncertainty = []

with open(directory+"/contactInEKF_est_data.txt", "r") as file:
    for line in file:
        name, time, data = parse_line(line)
        if name=='True_Rotation':
            contactInEKF_R_true.append(data)
        elif name=='Pred_Rotation':
            contactInEKF_R_pred_T.append(data)
        elif name=='Uncertainty':
            contactInEKF_uncertainty.append(data)
            t.append(time)

# Convert lists to Numpy arrays
contactInEKF_R_true = torch.tensor(contactInEKF_R_true, dtype=torch.float32).view(-1,3,3)
contactInEKF_R_pred_T = torch.tensor(contactInEKF_R_pred_T, dtype=torch.float32).view(-1,3,3)
contactInEKF_residual = pytorch3d.transforms.matrix_to_axis_angle(torch.matmul(contactInEKF_R_pred_T, contactInEKF_R_true)).detach().cpu().numpy()
contactInEKF_uncertainty = np.array(contactInEKF_uncertainty)
t = np.array(t)

# 영점 검정선
zero_line = np.zeros(t.shape[0])

# InEKF 검정 --, contactInEKF 초록 --, MLP 파랑, GRU 빨강

plt.figure(figsize=(10, 6))  # 그래프의 크기 설정

plt.subplot(231)
plt.plot(t, InEKF_uncertainty[:,0], 'orange', label='InEKF')
plt.plot(t, contactInEKF_uncertainty[:,0], 'g', label='contact_InEKF')
# plt.plot(t, mlp_uncertainty[:,0], 'b', label='mlp')
# plt.plot(t, gru_uncertainty[:,0], 'r', label='gru')
plt.legend()
plt.xlabel('time [s]')
plt.title("uncertainty x")
plt.grid(True)
# plt.ylim(-0.025,0.1)
# plt.ylim(9.5e-6,1.1e-5)

plt.subplot(232)
plt.plot(t, InEKF_uncertainty[:,1], 'orange', label='InEKF')
plt.plot(t, contactInEKF_uncertainty[:,1], 'g', label='contact_InEKF')
# plt.plot(t, mlp_uncertainty[:,1], 'b', label='mlp')
# plt.plot(t, gru_uncertainty[:,1], 'r', label='gru')
plt.legend()
plt.xlabel('time [s]')
plt.title("uncertainty y")
plt.grid(True)
# plt.ylim(-0.05, 0.2)
# plt.ylim(9.5e-6,1.1e-5)

plt.subplot(233)
plt.plot(t, InEKF_uncertainty[:,2], 'orange', label='InEKF')
plt.plot(t, contactInEKF_uncertainty[:,2], 'g', label='contact_InEKF')
# plt.plot(t, mlp_uncertainty[:,2], 'b', label='mlp')
# plt.plot(t, gru_uncertainty[:,2], 'r', label='gru')
plt.legend()
plt.xlabel('time [s]')
plt.title("uncertainty z")
plt.grid(True)
# plt.ylim(0.5,1.4)
# plt.ylim(0.00025,0.002)

plt.subplot(234)
plt.plot(t, zero_line, 'k')
plt.plot(t, InEKF_residual[:,0], 'orange', label='InEKF')
plt.plot(t, contactInEKF_residual[:,0], 'g', label='contact_InEKF')
# plt.plot(t, mlp_residual[:,0], 'b', label='mlp')
# plt.plot(t, gru_residual[:,0], 'r', label='gru')
plt.legend()
plt.legend(loc='lower right')
plt.xlabel('time [s]')
plt.ylabel('angle [rad]')
plt.ylim(-0.4,0.2)
plt.title("roll error")
plt.grid(True)

plt.subplot(235)
plt.plot(t, zero_line, 'k')
plt.plot(t, InEKF_residual[:,1], 'orange', label='InEKF')
plt.plot(t, contactInEKF_residual[:,1], 'g', label='contact_InEKF')
# plt.plot(t, mlp_residual[:,1], 'b', label='mlp')
# plt.plot(t, gru_residual[:,1], 'r', label='gru')
plt.legend()
plt.legend(loc='lower right')
plt.xlabel('time [s]')
plt.ylabel('angle [rad]')
plt.ylim(-0.4,0.2)
plt.title("pitch error")
plt.grid(True)

plt.subplot(236)
plt.plot(t, InEKF_residual[:,2], 'orange', label='InEKF')
plt.plot(t, contactInEKF_residual[:,2], 'g', label='contact_InEKF')
# plt.plot(t, mlp_residual[:,2], 'b', label='mlp')
# plt.plot(t, gru_residual[:,2], 'r', label='gru')
plt.legend()
plt.legend(loc='lower right')
plt.xlabel('time [s]')
plt.ylabel('angle [rad]')
plt.title("yaw error")
plt.ylim(-2,5)
plt.grid(True)


plt.tight_layout()
plt.show()