import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10, CIFAR100

cifar_10_root = '/wang_hp/zhy/data'
cifar_100_root = '/wang_hp/zhy/data'
    
####my_cifar_dataset
class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)

class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx, uq_idx

    def __len__(self):
        return len(self.targets)

####my_cifar_dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None

def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    # train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    # train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    # val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    # val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    whole_test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset = subsample_classes(deepcopy(whole_test_dataset), include_classes=train_classes)

    # # Either split train into train and val or use test set as val
    # train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    # val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    # all_datasets = {
    #     'train_labelled': train_dataset_labelled,
    #     'train_unlabelled': train_dataset_unlabelled,
    #     'val': val_dataset_labelled,
    #     'test': test_dataset,
    # }

    return train_dataset_labelled, test_dataset, train_dataset_unlabelled

def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                       prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    # train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    # train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    # val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    # val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    whole_test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)
    test_dataset = subsample_classes(deepcopy(whole_test_dataset), include_classes=train_classes)

    # Either split train into train and val or use test set as val
    # train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    # val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    # all_datasets = {
    #     'train_labelled': train_dataset_labelled,
    #     'train_unlabelled': train_dataset_unlabelled,
    #     'val': val_dataset_labelled,
    #     'test': test_dataset,
    # }

    return train_dataset_labelled, test_dataset, train_dataset_unlabelled

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)

# if __name__ == '__main__':

#     x = get_cub_datasets(None, None, split_train_val=False,
#                          train_classes=range(100), prop_train_labels=0.5)

#     print('Printing lens...')
#     for k, v in x.items():
#         if v is not None:
#             print(f'{k}: {len(v)}')

#     print('Printing labelled and unlabelled overlap...')
#     print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
#     print('Printing total instances in train...')
#     print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

#     print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
#     print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
#     print(f'Len labelled set: {len(x["train_labelled"])}')
#     print(f'Len unlabelled set: {len(x["train_unlabelled"])}')