import os
import cv2
import numpy as np
import random
import blobfile as bf
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

def _list_image_paths_recursively(data_dir, shuffle = False):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_paths_recursively(full_path))
    if shuffle is True:
        random.shuffle(results)
    return results


class CPDataset(Dataset):
    def __init__(self, source_data, target_data, dim, normed, image_size=(256, 256)):
        self.source_data = source_data
        self.target_data = target_data
        self.dim = dim
        self.normed = normed

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.crop((
                (x.width - min(x.size)) // 2,
                (x.height - min(x.size)) // 2,
                (x.width + min(x.size)) // 2,
                (x.height + min(x.size)) // 2
            ))),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        source_path = self.source_data[idx]
        target_path = self.target_data[idx]

        source_file_name = os.path.basename(source_path).split(".")[0]
        target_file_name = os.path.basename(target_path).split(".")[0]

        if source_path.endswith('.npy'):
            source = np.load(source_path)
            target = np.load(target_path)

            if self.dim == 1:
                source = np.dot(source[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
                target = np.dot(target[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

            if self.normed is False:
                source = (source / 127.5) - 1.0
                target = (target / 127.5) - 1.0

            source = np.expand_dims(source, axis=0)
            target = np.expand_dims(target, axis=0)
        elif source_path.endswith('.png') or source_path.endswith('.jpeg')  or source_path.endswith('.jpg'):
            source = cv2.imread(source_path).astype(np.float32)
            target = cv2.imread(target_path).astype(np.float32)

            if self.dim == 1:
                source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
                target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

            source = torch.tensor(source, dtype=torch.float32) / 255.0
            target = torch.tensor(target, dtype=torch.float32) / 255.0

            if self.dim == 3:
                source = source.permute(2, 0, 1)
                target = target.permute(2, 0, 1)

            source = self.transform(source)
            target = self.transform(target)

            if self.normed is False:
                source = (source * 2.0) - 1.0
                target = (target * 2.0) - 1.0

        return source, target, source_file_name, target_file_name


def GetDataset(phase = "test", input_path = "/dataset", contrast1 = 'T1', contrast2 = 'T2', shuffle = False, dim = 1, normed=False, unaligned=False):
    if phase == "train_cpn" or phase == "train_model":
        phase = "train"
    elif phase == "val":
        phase = "val"
    else:
        phase = "test"
    source_path = os.path.join(input_path, phase, contrast1)
    target_path = os.path.join(input_path, phase, contrast2)

    source_data = _list_image_paths_recursively(source_path, shuffle)
    target_data = _list_image_paths_recursively(target_path, shuffle)

    dataset = CPDataset(source_data, target_data, dim, normed)
    return dataset
