import os
import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent

b_nx, b_ny, b_nz = 64, 64, 64
cover_ratio = 0.5
pad_nx, pad_ny, pad_nz = b_nx // 4, b_ny // 4, b_nz // 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Divide paired 3D volumes and labels into overlapping training patches."
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "images",
        help="Directory containing input image volumes in NIfTI format."
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "labels",
        help="Directory containing input label volumes in NIfTI format."
    )
    parser.add_argument(
        "--image_output_dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "images_patches",
        help="Directory for saving image patches."
    )
    parser.add_argument(
        "--label_output_dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "labels_patches",
        help="Directory for saving label patches."
    )
    return parser.parse_args()


args = parse_args()

image_folder = args.image_dir.resolve()
label_folder = args.label_dir.resolve()
image_patch_save_folder = args.image_output_dir.resolve()
label_patch_save_folder = args.label_output_dir.resolve()

if not os.path.exists(image_patch_save_folder):
    os.makedirs(image_patch_save_folder)
if not os.path.exists(label_patch_save_folder):
    os.makedirs(label_patch_save_folder)
image_names = sorted(os.listdir(image_folder))
label_names = sorted(os.listdir(label_folder))
for idx in range(len(image_names)):
    print("********************************************************")
    print("Now is processing: ", image_names[idx])
    image_path = image_folder / image_names[idx]
    label_path = label_folder / label_names[idx]
    image_sitk = sitk.ReadImage(str(image_path))
    label_sitk = sitk.ReadImage(str(label_path))
    image_array = sitk.GetArrayFromImage(image_sitk)
    label_array = sitk.GetArrayFromImage(label_sitk)
    label_array[label_array != 1] = 0
    label_array[label_array == 1] = 1
    v_nx = image_array.shape[2]
    v_ny = image_array.shape[1]
    v_nz = image_array.shape[0]

    st_nx = np.int32(np.round(b_nx - cover_ratio * b_nx))
    st_ny = np.int32(np.round(b_ny - cover_ratio * b_ny))
    st_nz = np.int32(np.round(b_nz - cover_ratio * b_nz))

    blks_nx = np.int32(np.floor((v_nx - b_nx) / st_nx) + 1)
    blks_ny = np.int32(np.floor((v_ny - b_ny) / st_ny) + 1)
    blks_nz = np.int32(np.floor((v_nz - b_nz) / st_nz) + 1)

    print("X direction block number: ", blks_nx)
    print("Y direction block number: ", blks_ny)
    print("Z direction block number: ", blks_nz)

    current_patch_id = 0
    for z_idx in np.arange(0, blks_nz):
        z_start = np.min((z_idx * st_nz, v_nz - b_nz))
        for y_idx in np.arange(0, blks_ny):
            y_start = np.min((y_idx * st_ny, v_ny - b_ny))
            for x_idx in np.arange(0, blks_nx):
                x_start = np.min((x_idx * st_nx, v_nx - b_nx))
                image_patch = image_array[z_start: z_start + b_nz, y_start: y_start + b_ny, x_start: x_start + b_nx]
                label_patch = label_array[z_start: z_start + b_nz, y_start: y_start + b_ny, x_start: x_start + b_nx]
                if len(np.unique(label_patch)) > 1:
                    np.save(image_patch_save_folder / ("em710_train_pos_" + str(idx).rjust(4, '0') + "_" + str(current_patch_id).rjust(4, '0') + '.npy'), image_patch)
                    np.save(label_patch_save_folder / ("em710_train_pos_" + str(idx).rjust(4, '0') + "_" + str(current_patch_id).rjust(4, '0') + '.npy'), label_patch)
                elif np.random.random_sample() > 0.95:
                    np.save(image_patch_save_folder / ("em710_train_neg_" + str(idx).rjust(4, '0') + "_" + str(current_patch_id).rjust(4, '0') + '.npy'), image_patch)
                    np.save(label_patch_save_folder / ("em710_train_neg_" + str(idx).rjust(4, '0') + "_" + str(current_patch_id).rjust(4, '0') + '.npy'), label_patch)
                current_patch_id = current_patch_id + 1