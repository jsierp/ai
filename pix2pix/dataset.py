import torch.utils.data as data
from torchvision import transforms
from typing import List, Tuple
import cv2
import numpy as np


def zero_center(image):
    return image * 2 - 1


class Pix2PixDataset(data.Dataset):
    def __init__(
        self,
        input_paths: List[str],
        target_paths: List[str],
        image_size: int,
        *args,
        **kwargs
    ):
        super(Pix2PixDataset, self).__init__(*args, **kwargs)
        self.input_paths = input_paths
        self.target_paths = target_paths
        assert len(self.input_paths) == len(self.target_paths)

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                zero_center,
            ]
        )

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        x = cv2.imread(self.input_paths[index], cv2.IMREAD_UNCHANGED)
        y = cv2.cvtColor(
            cv2.imread(self.target_paths[index], cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        )
        x = self.transforms(x)
        y = self.transforms(y)
        return x, y
