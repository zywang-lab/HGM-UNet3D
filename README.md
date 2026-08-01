# HGM-UNet3D

## Project Overview

This project implements HGM-UNet3D, a three-dimensional deep learning framework for automatic bubble plume segmentation from multibeam water column data.

The project provides a complete pipeline including data preprocessing, dataset construction, model training, inference, and segmentation result visualization.

HGM-UNet3D is designed to exploit the intrinsic three-dimensional structural information of bubble plumes and improve segmentation performance under low target-to-background contrast and complex acoustic interference conditions.

## Quick Start

### 0. Data Preparation

Prepare the multibeam water column dataset.

The input data should be stored in NIfTI (`.nii.gz`) format. Organize the dataset according to the following structure:
```
data/
├── images/
│ ├── 001.nii.gz
│ ├── 002.nii.gz
│ ├── 003.nii.gz
│ └── ...
│
└── labels/
├── 001.nii.gz
├── 002.nii.gz
├── 003.nii.gz
└── ...
```
Note: Ensure that the image volumes and annotation masks are spatially aligned and have identical dimensions.

### 1. Build Dataset

Use `padding_and_make_patch.py` to divide the paired multibeam water column volumes and voxel-level annotation masks into overlapping 3D patches for model training.

Before running the script, edit the following paths in `padding_and_make_patch.py`:

```python
image_folder = "path/to/data/images/"
label_folder = "path/to/data/labels/"
image_patch_save_folder = "path/to/dataset/images_patches/"
label_patch_save_folder = "path/to/dataset/labels_patches/"
```
The patch size and overlap ratio can also be configured in the script:
```
b_nx, b_ny, b_nz = 64, 64, 64
cover_ratio = 0.5
```
Run the dataset construction script:
python padding_and_make_patch.py

#### Parameter Description

- `image_folder`: Directory containing the input multibeam water-column volumes in NIfTI (`.nii.gz`) format.
- `label_folder`: Directory containing the corresponding voxel-level bubble plume annotation masks in NIfTI (`.nii.gz`) format.
- `image_patch_save_folder`: Directory used to save the generated image patches in NumPy (`.npy`) format.
- `label_patch_save_folder`: Directory used to save the generated label patches in NumPy (`.npy`) format.
- `b_nx`, `b_ny`, `b_nz`: Dimensions of the extracted 3D patches along the X, Y, and Z directions. The default patch size is `64 × 64 × 64` voxels.
- `cover_ratio`: Overlap ratio between adjacent patches. The default value is `0.5`, corresponding to a sliding-window stride of `32 × 32 × 32` voxels.

The annotation masks are converted into binary masks, where:

0 represents the background;
1 represents the bubble plume.

All patches containing bubble plume voxels are retained. To reduce the imbalance between foreground and background samples, only a randomly selected subset of background-only patches is retained.

Output Structure

The generated dataset is saved as NumPy (.npy) arrays with the following structure:
```
dataset/
├── images_patches/
│   ├── em710_train_pos_0000_0000.npy
│   ├── em710_train_pos_0000_0001.npy
│   ├── em710_train_neg_0000_0002.npy
│   └── ...
│
└── labels_patches/
    ├── em710_train_pos_0000_0000.npy
    ├── em710_train_pos_0000_0001.npy
    ├── em710_train_neg_0000_0002.npy
    └── ...
```
Files containing pos correspond to patches that contain bubble plume voxels, whereas files containing neg correspond to background-only patches.

Note: Ensure that the image volumes and annotation masks have matching file names, identical dimensions, and consistent spatial alignment before running the script.
```
python padding_and_make_patch.py \
    --image_dir data/images \
    --label_dir data/labels \
    --image_output_dir dataset/images_patches \
    --label_output_dir dataset/labels_patches \
    --patch_size 64 64 64 \
    --overlap_ratio 0.5 \
    --negative_keep_ratio 0.05
```
### 2. Model Training

#### 2.1 Configure Training Parameters

Before training, edit the training script to specify the dataset directories, model output directory, and training hyperparameters.

Example configuration:

```python
# Training epochs
epochs = 800

# Batch size and gradient accumulation
real_batch_size = 8
accumulation_steps = 4
effective_batch_size = real_batch_size * accumulation_steps

# Learning-rate configuration
init_learning_rate = 0.0005
learning_rate_patience = 12
learning_rate_factor = 0.5

# Dataset and model output directories
image_patch_folder = "path/to/dataset/images_patches/"
label_patch_folder = "path/to/dataset/labels_patches/"
modelsave_path = "path/to/model-log/hgm_unet3d/"

The HGM-UNet3D model is initialized with one input channel and one output channel:
model = EViT_ResUNet3D(
    input_ch=1,
    output_ch=1,
    init_feats=16
)
The optimizer and loss function are configured as follows:
optimizer = optim.Adam(
    model.parameters(),
    lr=init_learning_rate,
    weight_decay=0.0001,
    eps=1e-5
)

loss_function = DiceLoss(
    include_background=False,
    sigmoid=True
)
A ReduceLROnPlateau scheduler is used to reduce the learning rate when the validation loss stops improving:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    patience=learning_rate_patience,
    factor=learning_rate_factor
)
```
#### 2.2 Training Parameter Description

- `epochs`: Total number of training epochs.
- `real_batch_size`: Training batch size (adjust according to GPU memory).
- `accumulation_steps`: Number of gradient accumulation steps.
- `effective_batch_size`: Effective training batch size after gradient accumulation.
- `init_learning_rate`: Initial learning rate.
- `learning_rate_patience`: Number of epochs with no improvement in validation loss before reducing the learning rate.
- `learning_rate_factor`: Learning-rate reduction factor.
- `image_patch_folder`: Directory containing the training image patches.
- `label_patch_folder`: Directory containing the corresponding training label patches.
- `modelsave_path`: Model output directory.

The training dataset uses data augmentation, whereas the validation dataset is loaded without augmentation.

#### 2.3 Start Training
Run the training script:

```
python train.py

```
####2.4 Model Save Structure
Training records are saved as Excel (.xlsx) files, and model parameters are saved as PyTorch (.pth) checkpoints.

The output directory has the following structure:
```
model-log/
└── hgm_unet3d/
    ├── UNet3D_000001.xlsx
    ├── UNet3D_000001.pth
    ├── UNet3D_000002.xlsx
    ├── UNet3D_000003.xlsx
    ├── ...
    └── UNet3D_000011.pth
```
An Excel file is generated after each training epoch to record the training and validation losses, segmentation metrics, and current learning rate.

Model checkpoints are saved periodically during training.

## 3. Model Inference

### 3.1 Single Checkpoint Inference

Edit the `predict_single_file.py` file:

```python
data_path = "data/test/EM302_test.nii.gz"
pred_path = "output/EM302_prediction.nii.gz"

b_nx, b_ny, b_nz = 64, 64, 64
st_nx, st_ny, st_nz = 32, 32, 32
pad_nx, pad_ny, pad_nz = 16, 16, 16

model.load_state_dict(
    torch.load(
        "model-log/hgm_unet3d_model/HGM_UNet3D_000xxx.pth",
        weights_only=True,
        map_location="cuda:0"
    )
)
```
Run inference:
```
python predict_single_file.py
```
Inference results are saved at the path specified by pred_path:
```
output/
└── EM302_prediction.nii.gz
```
























