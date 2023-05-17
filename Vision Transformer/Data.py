import os
from PIL import Image
from torch.utils.data import Dataset

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, 'leftImg8bit')
        self.label_dir = os.path.join(self.root_dir, 'gtFine')

        self.image_filenames = os.listdir(self.image_dir)

    def __getitem__(self, index):
        image_name = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_name)
        label_name = image_name.replace("leftImg8bit", "gtFine_labelIds")
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.image_filenames)

