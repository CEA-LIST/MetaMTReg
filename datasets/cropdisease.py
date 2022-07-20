# @copyright CEA-LIST/DIASI/SIALV/LVA (2022)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import os

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from PIL import Image

class CropDiseaseClassDataset(ClassDataset):

    folder = os.path.join('CropDisease','dataset','train')
    
    def __init__(self, root, transform=None, class_augmentations=None):
        super(CropDiseaseClassDataset, self).__init__(meta_train=False,
            meta_val=False, meta_test=True, meta_split=None,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self._labels = None
        
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CropDiseaseDataset(index, class_name, self.root,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def labels(self):
        if self._labels is None:
            self._labels = os.listdir(self.root)
        return self._labels

class CropDiseaseDataset(Dataset):
    def __init__(self, index, class_name, root,
             transform=None, target_transform=None):
        super(CropDiseaseDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.root = root
        self.class_name = class_name
        self.images = os.listdir(os.path.join(self.root, self.class_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.class_name, self.images[index])
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)

class CropDisease(CombinationMetaDataset):
    """
    Parameters
    ----------
    root : string
        Root directory where the dataset folder `CropDisease` exists.
    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way" 
        classification.
    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.
    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.
    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.
    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=True, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CropDiseaseClassDataset(root, transform=transform,
                                          class_augmentations=class_augmentations)
        super(CropDisease, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)