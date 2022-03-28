from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SnekData(Dataset):
    '''Snake dataset'''
    
    def __init__(self, csv_file, root_dir, transform=None):
        metadata = pd.read_csv(csv_file)
        self.classes = metadata['binomial_name'].unique()
        self.class_to_idx = self._map_classes(metadata,'class_id')
        self.country_to_idx = self._map_classes(metadata,'country')
        self.code_to_idx = self._map_classes(metadata,'code')
        self.metadata = metadata
        self.root = root_dir
        self.transform = transform

    def _map_classes(self, metadata, label):
        ids = list(metadata[label].unique())
        id_to_idx = {ids[i] : i for i in range(len(ids))}
        return id_to_idx

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.root,
                                self.metadata["file_path"].iloc[idx])
        image = Image.open(img_path).convert('RGB')
        sample = {
            'image' : image,
            'label_id' : self.class_to_idx[self.metadata["class_id"].iloc[idx]],
            'class_id' : self.metadata["class_id"].iloc[idx],
            'observation_id' : self.metadata["observation_id"].iloc[idx],
            'endemic' : int(self.metadata["endemic"].iloc[idx]),
            'country' : self.country_to_idx[self.metadata["country"].iloc[idx]],
            'code' : self.code_to_idx[self.metadata["code"].iloc[idx]]
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample


'''
class SnekData(Dataset):
    def __init__(self, imagefolder, indices):
        self.dataset = imagefolder #torch.utils.data.Subset(imagefolder, indices)
        self.indices = indices
        self.classes = imagefolder.classes
        self.class_to_idx = imagefolder.class_to_idx

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

'''
'''
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class SnekData(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        root (str): The dataset root directory
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, root, indices, loader=default_loader):
        super(SnekData, self).__init__(root)

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = IMG_EXTENSIONS
        self.classes = classes
        self.class_to_idx = class_to_idx
        
        self.dataset = torch.utils.data.Subset(self, indices)
        self.samples = samples
        self.labels = [x[1] for x in samples]
        
        
        self.imgs = self.samples

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        classes = self.classes[idx]
        return (image, classes)

    def __len__(self):
        return len(self.classes)
'''
