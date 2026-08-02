import os
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# DataProcess.py 
# HGM-UNet3D/data_processing/
SCRIPT_DIR = Path(__file__).resolve().parent

# HGM-UNet3D/
REPO_ROOT = SCRIPT_DIR.parent


b_nx, b_ny, b_nz = 64, 64, 64
cover_ratio = 0.5

pad_nx, pad_ny, pad_nz = b_nx // 4, b_ny // 4, b_nz // 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Divide paired 3D volumes and labels into "
            "overlapping training patches."
        )
    )

    parser.add_argument(
        "--image_dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "images",
        help="Directory containing input image volumes in NIfTI format."
    )

    parser.add_argument(
        "--label_dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "labels",
        help="Directory containing input label volumes in NIfTI format."
    )

    parser.add_argument(
        "--image_output_dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "images_patches",
        help="Directory for saving image patches."
    )

    parser.add_argument(
        "--label_output_dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "labels_patches",
        help="Directory for saving label patches."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_folder = args.image_dir.resolve()
    label_folder = args.label_dir.resolve()
    image_patch_save_folder = args.image_output_dir.resolve()
    label_patch_save_folder = args.label_output_dir.resolve()

    if not image_folder.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_folder}"
        )

    if not label_folder.exists():
        raise FileNotFoundError(
            f"Label directory does not exist: {label_folder}"
        )

    image_patch_save_folder.mkdir(parents=True, exist_ok=True)
    label_patch_save_folder.mkdir(parents=True, exist_ok=True)

    image_names = sorted(
        file_name
        for file_name in os.listdir(image_folder)
        if (image_folder / file_name).is_file()
    )

    label_names = sorted(
        file_name
        for file_name in os.listdir(label_folder)
        if (label_folder / file_name).is_file()
    )

    if len(image_names) != len(label_names):
        raise ValueError(
            "The number of image volumes does not match the number of "
            f"label volumes: {len(image_names)} images and "
            f"{len(label_names)} labels."
        )

    print(f"Image directory: {image_folder}")
    print(f"Label directory: {label_folder}")
    print(f"Image patch output: {image_patch_save_folder}")
    print(f"Label patch output: {label_patch_save_folder}")

    for idx in range(len(image_names)):
        print("********************************************************")
        print("Now processing:", image_names[idx])

        image_path = image_folder / image_names[idx]
        label_path = label_folder / label_names[idx]

        image_sitk = sitk.ReadImage(str(image_path))
        label_sitk = sitk.ReadImage(str(label_path))

        image_array = sitk.GetArrayFromImage(image_sitk)
        label_array = sitk.GetArrayFromImage(label_sitk)

        if image_array.shape != label_array.shape:
            raise ValueError(
                f"Shape mismatch for pair {idx}: "
                f"image {image_names[idx]} has shape {image_array.shape}, "
                f"but label {label_names[idx]} has shape "
                f"{label_array.shape}."
            )

        # Convert the annotation into a binary mask
        label_array = (label_array == 1).astype(np.uint8)

        v_nz, v_ny, v_nx = image_array.shape

        if v_nx < b_nx or v_ny < b_ny or v_nz < b_nz:
            raise ValueError(
                f"Volume {image_names[idx]} has shape "
                f"{image_array.shape}, which is smaller than the patch "
                f"size {(b_nz, b_ny, b_nx)}."
            )

        st_nx = int(round(b_nx * (1 - cover_ratio)))
        st_ny = int(round(b_ny * (1 - cover_ratio)))
        st_nz = int(round(b_nz * (1 - cover_ratio)))

        blks_nx = int(np.floor((v_nx - b_nx) / st_nx) + 1)
        blks_ny = int(np.floor((v_ny - b_ny) / st_ny) + 1)
        blks_nz = int(np.floor((v_nz - b_nz) / st_nz) + 1)

        print("X direction block number:", blks_nx)
        print("Y direction block number:", blks_ny)
        print("Z direction block number:", blks_nz)

        current_patch_id = 0
        positive_patch_count = 0
        negative_patch_count = 0

        for z_idx in range(blks_nz):
            z_start = min(z_idx * st_nz, v_nz - b_nz)

            for y_idx in range(blks_ny):
                y_start = min(y_idx * st_ny, v_ny - b_ny)

                for x_idx in range(blks_nx):
                    x_start = min(x_idx * st_nx, v_nx - b_nx)

                    image_patch = image_array[
                        z_start:z_start + b_nz,
                        y_start:y_start + b_ny,
                        x_start:x_start + b_nx
                    ]

                    label_patch = label_array[
                        z_start:z_start + b_nz,
                        y_start:y_start + b_ny,
                        x_start:x_start + b_nx
                    ]

                    patch_id = (
                        f"{idx:04d}_{current_patch_id:04d}"
                    )

                    # Save every patch containing plume voxels
                    if np.any(label_patch == 1):
                        patch_name = f"em_train_pos_{patch_id}.npy"

                        np.save(
                            image_patch_save_folder / patch_name,
                            image_patch
                        )
                        np.save(
                            label_patch_save_folder / patch_name,
                            label_patch
                        )

                        positive_patch_count += 1

                    # Randomly retain approximately 5% of background patches
                    elif np.random.random_sample() > 0.95:
                        patch_name = f"em_train_neg_{patch_id}.npy"

                        np.save(
                            image_patch_save_folder / patch_name,
                            image_patch
                        )
                        np.save(
                            label_patch_save_folder / patch_name,
                            label_patch
                        )

                        negative_patch_count += 1

                    current_patch_id += 1

        print(f"Saved positive patches: {positive_patch_count}")
        print(f"Saved negative patches: {negative_patch_count}")

    print("Data processing completed.")


if __name__ == "__main__":
    main()