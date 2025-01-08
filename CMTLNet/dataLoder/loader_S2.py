import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
import random
# from dataload_utils import (default_loader, default_loader_test)
from dataLoder.dataload_utils import (default_loader, default_loader_test)

np.random.seed(1337)
random.seed(1337)

def get_fold_data(fold_number: int, mode: str, pickle_path: str) -> Tuple[list, list]:
    """
    Load data for a specific fold and mode from a PKL file.

    Parameters:
    - mode (str): Either 'train' or 'test'.
    - fold_number (int): An integer between 0 and 4 indicating the fold index.
    - pickle_path (str): The file path to the PKL file.

    Returns:
    - Tuple containing:
        - paths (list): List of file paths for the specified fold.
        - class_labels (list): List of class labels corresponding to the file paths.

    Raises:
    - ValueError: If mode is not 'train' or 'test', or if fold_number is out of range.
    - FileNotFoundError: If the PKL file does not exist.
    - ValueError: If the PKL file cannot be unpickled.
    - KeyError: If the expected data structure is not found in the PKL file.
    """
    # Validate mode
    if mode not in ['train', 'test']:
        raise ValueError("mode must be 'train' or 'test'")

    # Validate fold_number
    if not isinstance(fold_number, int) or not (0 <= fold_number <= 4):
        raise ValueError("fold_number must be an integer between 0 and 4")

    # Attempt to load data from the PKL file
    try:
        with open(pickle_path, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"PKL file not found: {pickle_path}")
    except pickle.UnpicklingError:
        raise ValueError(f"Unable to load PKL file: {pickle_path}")

    # Construct the fold key
    fold_key = f'fold{fold_number}'

    # Retrieve data for the specified mode and fold
    try:
        fold_data = data[mode][fold_key]
        return fold_data['path'], fold_data['cls_label']
    except KeyError:
        raise KeyError(f"Missing {mode} data for {fold_key} in the data structure")

def split_path_by_label(file_paths, file_labels):
    file_paths_0 = []
    file_paths_1 = []
    for i in range(len(file_labels)):
        if file_labels[i]==0:
            file_paths_0.append(file_paths[i])
        elif file_labels[i]==1:
            file_paths_1.append(file_paths[i])
    return file_paths_0, file_paths_1


class TrainDataLoader_balance(Dataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        mode: str = 'train',
        fold: int = 1,
        pickle_path: str = ''
    ):
        """
        Initialize the training data loader.

        Parameters:
        - transform (Callable, optional): Transformation to apply to the images and masks.
        - mode (str): Mode of data loading, default is 'train'.
        - fold (int): Fold number to load data from..
        - pickle_path (str): Path to the PKL file containing data.
        """
        self.mode = mode
        self.transform = transform

        # Load file paths and labels for the specified fold and mode
        file_paths, file_labels = get_fold_data(fold, mode, pickle_path)
        self.file_paths_0, self.file_paths_1 = split_path_by_label(file_paths, file_labels)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return max(len(self.file_paths_0), len(self.file_paths_1))

    def get_match_tensor(
        self,
        images: list,
        masks: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate matched tensors from images and masks.

        Parameters:
        - images (list): List of image tensors.
        - masks (list): List of mask tensors.

        Returns:
        - Tuple containing:
            - IT1 (Tensor)
        """
        # Apply transformations if provided
        if self.transform is not None:
            images = [self.transform(img) for img in images]
            masks = [self.transform(mask) for mask in masks]

        # Unpack masks
        mask_original, mask_positive, mask_negative = masks

        # Apply original mask to images and concatenate along a new dimension
        images_tensor = torch.cat([
            (images[0] * mask_original).unsqueeze(0),
            (images[1] * mask_original).unsqueeze(0)
        ], dim=0)

        # Create various tensor combinations
        IT1 = torch.cat([images_tensor, mask_original.unsqueeze(0)], dim=0)
        return IT1.float()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset.

        Parameters:
        - index (int): Index of the sample to retrieve.

        Returns:
        - Tuple containing:
            - IT1 (Tensor): Image tensors.
            - class_label_tensor (Tensor): class labels.
        """
        index_0 = index % len(self.file_paths_0)
        random_index = random.randint(0, len(self.file_paths_1))
        index_1 = (index + random_index) % len(self.file_paths_1)

        # Load images and masks using a default loader
        images_0, masks_0 = default_loader(self.file_paths_0[index_0])
        images_1, masks_1 = default_loader(self.file_paths_1[index_1])

        # Generate matched tensors
        IT1_0 = self.get_match_tensor(images_0, masks_0)
        IT1_1 = self.get_match_tensor(images_1, masks_1)

        images_combined = torch.cat([IT1_0.unsqueeze(0), IT1_1.unsqueeze(0)], dim=0)
        labels_combined = torch.tensor([0, 1], dtype=torch.long).unsqueeze(0)
        return images_combined, labels_combined

class TrainDataLoader(Dataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        mode: str = 'train',
        fold: int = 1,
        pickle_path: str = ''
    ):
        """
        Initialize the training data loader.

        Parameters:
        - transform (Callable, optional): Transformation to apply to the images and masks.
        - mode (str): Mode of data loading, default is 'train'.
        - fold (int): Fold number to load data from..
        - pickle_path (str): Path to the PKL file containing data.
        """
        self.mode = mode
        self.transform = transform

        # Load file paths and labels for the specified fold and mode
        self.file_paths, self.file_labels = get_fold_data(fold, mode, pickle_path)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.file_paths)

    def get_match_tensor(
        self,
        images: list,
        masks: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate matched tensors from images and masks.

        Parameters:
        - images (list): List of image tensors.
        - masks (list): List of mask tensors.

        Returns:
        - Tuple containing:
            - IT1 (Tensor)
        """
        # Apply transformations if provided
        if self.transform is not None:
            images = [self.transform(img) for img in images]
            masks = [self.transform(mask) for mask in masks]

        # Unpack masks
        mask_original, mask_positive, mask_negative = masks

        # Apply original mask to images and concatenate along a new dimension
        images_tensor = torch.cat([
            (images[0] * mask_original).unsqueeze(0),
            (images[1] * mask_original).unsqueeze(0)
        ], dim=0)

        # Create various tensor combinations
        IT1 = torch.cat([images_tensor, mask_original.unsqueeze(0)], dim=0)
        return IT1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset.

        Parameters:
        - index (int): Index of the sample to retrieve.

        Returns:
        - Tuple containing:
            - IT1 (Tensor): Image tensors.
            - class_label_tensor (Tensor): class labels.
        """
        # Load images and masks using a default loader
        images, masks = default_loader(self.file_paths[index])
        class_label = self.file_labels[index]
        # Generate matched tensors
        IT1 = self.get_match_tensor(images, masks)
        class_label_tensor = torch.tensor(class_label, dtype=torch.long)
        return IT1.float(), class_label_tensor

class TestDataLoader(Dataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        mode: str = 'test',
        fold: int = 1,
        random_flag: bool = False,
        pickle_path: str = ''
    ):
        """
        Initialize the testing data loader.

        Parameters:
        - transform (Callable, optional): Transformation to apply to the images and masks.
        - mode (str): Mode of data loading, default is 'test'.
        - fold (int): Fold number to load data from.
        - random_flag (bool): If True, randomly select tensors; otherwise, use all.
        - pickle_path (str): Path to the PKL file containing data.
        """
        self.random_flag = random_flag
        self.mode = mode
        self.transform = transform

        # Load file paths and labels for the specified fold and mode
        self.file_paths, self.file_labels = get_fold_data(fold, mode, pickle_path)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.file_paths)

    def get_match_tensor(
        self,
        images: list,
        masks: list
    ) -> torch.Tensor:
        """
        Generate matched tensor from images and masks for testing.

        Parameters:
        - images (list): List of image tensors.
        - masks (list): List of mask tensors.

        Returns:
        - IT1 (Tensor)
        """
        # Apply transformations if provided
        if self.transform is not None:
            images = [self.transform(img) for img in images]
            masks = [self.transform(mask) for mask in masks]

        # Use only the original mask for testing
        mask_original = masks[0]

        # Apply original mask to images and concatenate along a new dimension
        images_tensor = torch.cat([
            (images[0] * mask_original).unsqueeze(0),
            (images[1] * mask_original).unsqueeze(0)
        ], dim=0)

        # Create IT1 tensor
        IT1 = torch.cat([images_tensor, mask_original.unsqueeze(0)], dim=0)

        return IT1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the test dataset.

        Parameters:
        - index (int): Index of the sample to retrieve.

        Returns:
        - Tuple containing:
            - images_IT (Tensor): Image IT tensors.
            - class_label (Tensor): Class label.
        """
        # Load images and masks using a default test loader
        images, masks = default_loader_test(self.file_paths[index])
        class_label = self.file_labels[index]

        # Generate matched tensor
        images_IT = self.get_match_tensor(images, masks)

        # Convert class label to tensor
        class_label_tensor = torch.tensor(class_label, dtype=torch.long)

        return images_IT.float(), class_label_tensor.long()


if __name__ == '__main__':
    import monai
    from torch.utils.data import DataLoader

    pkl_path = './CMTLNet_Path/1p19q.pkl'
    pkl_path = './CMTLNet_Path/LHG.pkl'
    pkl_path = './CMTLNet_Path/IDH.pkl'

    # Define the transformation for the validation stage
    validation_transforms = monai.transforms.Compose([
        monai.transforms.ToTensor(),
    ])

    # Initialize the training dataset with transformation and data reading type
    training_dataset = TrainDataLoader(read_type='train', fold=0, transform=validation_transforms,
                                       random_flag=False, pkl_path=pkl_path)

    # Create the training data loader
    training_loader = TrainDataLoader_balance(training_dataset,
                                 batch_size=1,  # One sample per batch
                                 shuffle=True,  # Shuffle the data order
                                 drop_last=True,  # Drop the last incomplete batch
                                 num_workers=1,  # Number of worker threads for data loading
                                 pin_memory=False)  # Whether to pin memory for CUDA

    # Output the size of the training dataset
    print("Number of training samples: {}".format(len(training_dataset)))

    # Loop through each batch in the data loader
    for batch_data in training_loader:
        # Unpack the data for each batch
        imgs_IT, imgs_INT, cls_label = batch_data