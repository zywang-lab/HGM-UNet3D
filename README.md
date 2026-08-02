# HGM-UNet3D

## Project Overview

This project implements HGM-UNet3D, a three-dimensional deep learning framework for automatic bubble plume segmentation from multibeam water column data.

The project provides a computational pipeline including dataset construction, model training, checkpoint-based inference, and bubble plume volume estimation.

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

Use `DataProcess.py` to divide the paired multibeam water column volumes and voxel-level annotation masks into overlapping 3D patches for model training.

Run the dataset construction script using the default project structure:
The patch size and overlap ratio can also be configured in the script:
The default patch size is 64 × 64 × 64 voxels with an overlap ratio of 0.5.
Run the dataset construction script using the default project structure:

```bash
python data_processing/DataProcess.py
```

or specify custom paths:

```bash
python data_processing/DataProcess.py \
    --image_dir data_processing/data/images \
    --label_dir data_processing/data/labels \
    --image_output_dir dataset/images_patches \
    --label_output_dir dataset/labels_patches
```

#### Parameter Description

--image_dir
--label_dir
--image_output_dir
--label_output_dir

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

### 2. Model Training

#### 2.1 Configure Training Parameters

Run the training script using the default project structure:

```bash
python train/train.py
```

or specify custom paths:

```bash
python train/train.py \
    --image_dir dataset/images_patches \
    --label_dir dataset/labels_patches \
    --output_dir model-log/hgm_unet3d
```

The HGM-UNet3D model is initialized with one input channel and one output channel:
```
model = HGM_UNet3D(
    input_ch=1,
    output_ch=1,
    init_feats=16
)
```
The optimizer and loss function are configured as follows:
```
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
```
A ReduceLROnPlateau scheduler is used to reduce the learning rate when the validation loss stops improving:
```
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
- --image_dir: Directory containing the training image patches.
- --label_dir: Directory containing the corresponding training label patches.
- --output_dir: Directory for saving training logs and model checkpoints.

The training dataset uses data augmentation, whereas the validation dataset is loaded without augmentation.

#### 2.3 Model Save Structure
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

### 3.1 Checkpoint Inference

Run inference using the default project structure:

```bash
python inference/inference.py
```

or specify custom paths:

```bash
python inference/inference.py \
    --input data/test/EM302_test.nii.gz \
    --checkpoint model-log/hgm_unet3d/your_checkpoint.pth \
    --output output/EM302_prediction.nii.gz
```
### 3.2 Bubble Plume Volume Estimation

Use the `calculate_volume.py` script to estimate the physical volume of segmented bubble plumes.

```bash
python calculate_volume.py
```

Edit the `calculate_volume.py` file to specify the predicted segmentation file:

```python
file_path = "output/EM302_prediction.nii.gz"
```

The script will automatically:

1. Load the predicted bubble plume segmentation in NIfTI (`.nii.gz`) format.
2. Calculate the number of segmented voxels.
3. Obtain the voxel size from the image header.
4. Estimate the physical bubble plume volume.
5. Output the calculated volume in different units, including `mm³`, `cm³`, `m³`, and `L`.

Example output:

```text
Target volume: 0.012345 m³

All units:
Volume: 12345678.00 mm³
Volume: 12345.68 cm³
Volume: 0.012346 m³
Volume: 12.346 L
Voxel count: 123456
Voxel size: (dx, dy, dz) mm
```






















