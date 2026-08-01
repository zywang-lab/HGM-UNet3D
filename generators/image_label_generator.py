import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import zoom, map_coordinates, binary_dilation, binary_erosion, gaussian_filter, \
    distance_transform_edt


class Image_Label_train(Dataset):


    def __init__(self, image_label_pairs):
        self.image_label_pairs = image_label_pairs
        self.original_shape = (64, 64, 64)

    def __getitem__(self, index):


        image_path = self.image_label_pairs[index][0]
        label_path = self.image_label_pairs[index][1]
        image_array = np.float32(np.load(image_path))
        label_array = np.float32(np.load(label_path))

        if np.random.random_sample() > 0.8:
            image_data = np.expand_dims(image_array, axis=0)
            label_data = np.expand_dims(label_array, axis=0)
            return torch.from_numpy(image_data), torch.from_numpy(label_data)
        else:
            if np.random.random_sample() > 0.6:
                image_array = np.flip(image_array, axis=0).copy()
                label_array = np.flip(label_array, axis=0).copy()
            if np.random.random_sample() > 0.6:
                image_array = np.flip(image_array, axis=1).copy()
                label_array = np.flip(label_array, axis=1).copy()
            if np.random.random_sample() > 0.6:
                image_array = np.flip(image_array, axis=2).copy()
                label_array = np.flip(label_array, axis=2).copy()

            if np.random.random_sample() > 0.5:
                k = np.random.randint(-3, 4)  # 随机选择旋转次数 (-3到3次，对应-270到270度)
                image_array = np.rot90(image_array, k, axes=(1, 2)).copy()  # 在H-W平面旋转
                label_array = np.rot90(label_array, k, axes=(1, 2)).copy()  # 确保标签与图像同步旋转

            if np.random.random_sample() > 0.9:
                scale = np.float32(np.random.uniform(low=0.9, high=1.1, size=1))  # 亮度缩放因子
                image_array = image_array * scale  # 调整图像亮度

            if np.random.random_sample() > 0.9:
                contrast_factor = np.random.uniform(0.8, 1.2)
                mean_val = np.mean(image_array)
                image_array = (image_array - mean_val) * contrast_factor + mean_val

            if np.random.random_sample() > 0.8:
                image_array, label_array = self._elastic_deformation(image_array, label_array)



            image_data = np.expand_dims(image_array, axis=0)
            label_data = np.expand_dims(label_array, axis=0)
            return torch.from_numpy(image_data), torch.from_numpy(label_data)

    def __len__(self):
        return len(self.image_label_pairs)

    def _elastic_deformation(self, image, label, alpha=4, sigma=2):

        shape = image.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha


        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))


        distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

        distorted_label = map_coordinates(label, indices, order=0, mode='constant').reshape(shape)

        return distorted_image, distorted_label


class Image_Label_valid(Dataset):

    def __init__(self, image_label_pairs):
        self.image_label_pairs = image_label_pairs

    def __getitem__(self, index):

        image_path = self.image_label_pairs[index][0]
        label_path = self.image_label_pairs[index][1]
        image_array = np.float32(np.load(image_path))
        label_array = np.float32(np.load(label_path))

        image_data = np.expand_dims(image_array, axis=0)
        label_data = np.expand_dims(label_array, axis=0)
        return torch.from_numpy(image_data), torch.from_numpy(label_data)

    def __len__(self):
        return len(self.image_label_pairs)