import os
import torch
import numpy as np
import SimpleITK as sitk
from monai.networks.nets import AttentionUnet

############################################################################
data_path = r"E:\Datasets\TestData\EM302_test.nii.gz"
pred_path = r"E:\Datasets\TestData\pred\Experimentation23\EM302_pred_att_final.nii.gz"
b_nx, b_ny, b_nz = 64, 64, 64
st_nx, st_ny, st_nz = 32, 32, 32
pad_nx, pad_ny, pad_nz = 16, 16, 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#############################################################################

model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
)
model = model.to(device)
model.load_state_dict(torch.load(r"E:\PycharmProjects\MSDSegEViT-3DResUNet\modelsave\AttentionUnet2\AttentionUNet_000241.pth", weights_only=True, map_location="cuda:0"))
model.eval()
#############################################################################

image_sitk = sitk.ReadImage(data_path)
image_array = sitk.GetArrayFromImage(image_sitk)

image_padded = np.pad(image_array, [(pad_nz, pad_nz), (pad_ny, pad_ny), (pad_nx, pad_nx)], mode="constant", constant_values=0)
v_nx = image_padded.shape[2]
v_ny = image_padded.shape[1]
v_nz = image_padded.shape[0]
blks_nx = np.int32(np.floor((v_nx - b_nx) / st_nx) + 1)
blks_ny = np.int32(np.floor((v_ny - b_ny) / st_ny) + 1)
blks_nz = np.int32(np.floor((v_nz - b_nz) / st_nz) + 1)
############################################################################
label_pred = np.zeros(shape=(image_array.shape[0], image_array.shape[1], image_array.shape[2]))
for z_idx in np.arange(0, blks_nz):
    z_start = np.min((z_idx * st_nz, v_nz - b_nz))
    z_start_pred = np.min((z_idx * b_nz / 2, label_pred.shape[0] - b_nz / 2))
    z_start_pred = int(z_start_pred)
    for y_idx in np.arange(0, blks_ny):
        y_start = np.min((y_idx * st_ny, v_ny - b_ny))
        y_start_pred = np.min((y_idx * b_ny / 2, label_pred.shape[1] - b_ny / 2))
        y_start_pred = int(y_start_pred)
        for x_idx in np.arange(0, blks_nx):
            x_start = np.min((x_idx * st_nx, v_nx - b_nx))
            x_start_pred = np.min((x_idx * b_nx / 2, label_pred.shape[2] - b_nx / 2))
            x_start_pred = int(x_start_pred)
            image_patch = image_padded[z_start: z_start + b_nz, y_start: y_start + b_ny, x_start: x_start + b_nx]
            image_patch = image_patch.astype(np.float32)
            image_patch = np.reshape(image_patch, [1, 1, b_nz, b_ny, b_nx])
            image_patch = torch.from_numpy(image_patch)
            image_patch = image_patch.to(device)
            pred_patch = model(image_patch)
            label_pred[z_start_pred: z_start_pred + int(b_nz / 2), y_start_pred: y_start_pred + int(b_ny / 2), x_start_pred: x_start_pred + int(b_nx / 2)] = pred_patch[0][0][16:48, 16:48, 16:48].cpu().detach().numpy()

label_pred[label_pred >= 0.5] = 1
label_pred[label_pred < 0.5] = 0
label_pred = np.uint8(label_pred)


label_pred_sitk = sitk.GetImageFromArray(label_pred)
label_pred_sitk.SetOrigin(image_sitk.GetOrigin())
label_pred_sitk.SetSpacing(image_sitk.GetSpacing())
label_pred_sitk.SetDirection(image_sitk.GetDirection())


sitk.WriteImage(label_pred_sitk, pred_path)


