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

Files containing pos correspond to patches that contain bubble plume voxels, whereas files containing neg correspond to background-only patches.

Note: Ensure that the image volumes and annotation masks have matching file names, identical dimensions, and consistent spatial alignment before running the script.

python padding_and_make_patch.py \
    --image_dir data/images \
    --label_dir data/labels \
    --image_output_dir dataset/images_patches \
    --label_output_dir dataset/labels_patches \
    --patch_size 64 64 64 \
    --overlap_ratio 0.5 \
    --negative_keep_ratio 0.05

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

#### 2.2 Training Parameter Description
epochs: Total number of training epochs. The default value is 800.
real_batch_size: Number of 3D patches processed by the GPU in each iteration. The default value is 8.
accumulation_steps: Number of iterations used for gradient accumulation. The default value is 4.
effective_batch_size: Effective batch size after gradient accumulation. With the default configuration, the effective batch size is 32.
init_learning_rate: Initial learning rate of the Adam optimizer. The default value is 0.0005.
learning_rate_patience: Number of epochs with no improvement in validation loss before reducing the learning rate.
learning_rate_factor: Factor used to reduce the learning rate. The default value is 0.5.
image_patch_folder: Directory containing the input image patches in NumPy (.npy) format.
label_patch_folder: Directory containing the corresponding binary label patches.
modelsave_path: Directory used to save model checkpoints and training records.
input_ch: Number of input channels. Multibeam water-column volumes are treated as single-channel data.
output_ch: Number of output channels. A single output channel is used for binary bubble plume segmentation.
init_feats: Number of initial feature channels in the network.
weight_decay: L2 regularization coefficient used by the Adam optimizer.
learning_rate_patience: Patience parameter of the learning-rate scheduler.
learning_rate_factor: Multiplicative factor applied when the learning rate is reduced.

The image–label patch pairs are randomly shuffled and divided into training and validation subsets using an 80:20 ratio.

The training dataset uses data augmentation, whereas the validation dataset is loaded without augmentation.

#### 2.3 Start Training
Run the training script:
python train.py
During training, the pipeline performs the following operations:

Loads paired image and label patches;
Randomly divides the patches into training and validation subsets;
Applies data augmentation to the training samples;
Performs forward propagation through HGM-UNet3D;
Computes the Dice loss;
Updates the model using gradient accumulation;
Evaluates the model on the validation subset;
Records segmentation metrics and training losses;
Saves model checkpoints and Excel training records.

The following segmentation metrics are calculated during training and validation:

Dice Similarity Coefficient (DSC);
Intersection over Union (IoU);
Volume Overlap Error (VOE);
Precision;
Recall;
Accuracy.
####2.4 Model Save Structure
Training records are saved as Excel (.xlsx) files, and model parameters are saved as PyTorch (.pth) checkpoints.

The output directory has the following structure:

model-log/
└── hgm_unet3d/
    ├── UNet3D_000001.xlsx
    ├── UNet3D_000001.pth
    ├── UNet3D_000002.xlsx
    ├── UNet3D_000003.xlsx
    ├── ...
    └── UNet3D_000011.pth

An Excel file is generated after each training epoch to record the training and validation losses, segmentation metrics, and current learning rate.

Model checkpoints are saved periodically during training.

### 3. Model Inference

#### 3.1 Configure Inference Parameters

Before inference, edit `inference.py` to specify the input volume, model checkpoint, and output path.

Example configuration:

```python
# Input multibeam water-column volume
data_path = "path/to/test_volume.nii.gz"

# Output segmentation result
pred_path = "path/to/output/prediction.nii.gz"

# Model checkpoint
checkpoint_path = "path/to/checkpoint/HGM_UNet3D_000241.pth"

# Sliding-window configuration
b_nx, b_ny, b_nz = 64, 64, 64
st_nx, st_ny, st_nz = 32, 32, 32
pad_nx, pad_ny, pad_nz = 16, 16, 16




























## Project Overview
This project implements a three-dimensional deep learning framework developed for automatic bubble plume segmentation from volumetric multibeam water column data. The framework is designed to exploit the intrinsic three-dimensional structures of bubble plumes and improve segmentation performance under complex acoustic interference conditions.
This repository provides:
- the HGM-UNet3D model implementation;
- inference pipeline;
- example multibeam water column volumes;
- visualization scripts for segmentation results.
## Quick Start

This section provides instructions for training, inference, and visualization of HGM-UNet3D.

## Introduction

HGM-UNet3D is an efficient and lightweight 3D deep learning framework for bubble plume segmentation in multibeam water column data.

The framework exploits volumetric information to improve the segmentation of complex bubble plume structures in multibeam water column images.

This repository is intended for demonstration purposes. During the manuscript review period, only selected examples and visualization scripts are provided.

## Demonstration

Example of 3D bubble plume segmentation on multibeam water column data.

<p align="center">
<img src="images\demo_result_readme.png" width="100%">
</p>
Left: Input MBES volume. Middle: Ground-truth annotation. Right: Prediction generated by HGM-UNet3D.


## Repository Structure

```text
HGM-UNet3D/
├── images/
│   └── Demonstration figures used in the README.
├── sample_data/
│   ├── demo_volume.nii.gz
│   ├── demo_ground_truth.nii.gz
│   └── demo_prediction.nii.gz
├── demo_visualization.ipynb
├── README.md
└── .gitignore
```

- **images/** – Figures used for visualization in the repository documentation.
- **sample_data/** – Example multibeam water column volume, ground-truth annotation, and corresponding segmentation result.
- **demo_visualization.ipynb** – Interactive notebook for loading, visualizing, and exploring the demonstration data.
- **README.md** – Project overview, repository description, and usage instructions.

## Project Status

The associated manuscript is currently under review.

To protect unpublished research results, this repository currently contains only demonstration materials.

Additional source code and documentation may be released after publication.

## Citation

The manuscript is currently under review.

Citation information will be updated after publication.
