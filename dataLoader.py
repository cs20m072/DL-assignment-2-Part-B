import os

from PIL import Image
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import configuration

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageInMemoryDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.images = []
        self.labels = []
        self.transforms = transform

        for image, label in tqdm(zip(data, labels), total=len(data)):
            img = Image.open(image)
            self.images.append(img.copy())
            self.labels.append(label)
            img.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        img_tensor = self.transforms(img)
        i = img_tensor.shape[0]
        if img_tensor.shape[0] == 1:
            img = img.convert("RGB")
            img_tensor = self.transforms(img)
        label = self.labels[item]
        return (img_tensor, label)


def get_file_list_with_labels(directory):
    # return list of image file names and corresponding labels from directory
    dir = os.path.dirname(__file__)
    dir = dir + directory
    data = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if (f.endswith('jpg')):
                data.append(root + '/' + f)
                s = root[root.rfind('/') + 1:]
                labels.append(configuration.labels.index(s))
    return data, labels


# return training and validation data loaders
def get_training_data_loaders():
    # load dataset file names in list with their corresponding labels
    data, labels = get_file_list_with_labels(configuration.training_directory)

    train_x, val_x, train_y, val_y = train_test_split(data, labels, test_size=0.1)

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=[configuration.image_width, configuration.image_height]),
            transforms.Normalize((0,), (1,))])

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=[configuration.image_width, configuration.image_height]),
            transforms.Normalize((0,), (1,))])

    if configuration.augment_data:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),
                transforms.Resize(size=[configuration.image_width, configuration.image_height]),
                transforms.Normalize((0,), (1,))])

    training_dataset = ImageInMemoryDataset(train_x, train_y, train_transform)
    validation_dataset = ImageInMemoryDataset(val_x, val_y, test_transform)

    training_dataloader = DataLoader(training_dataset, configuration.batch_size, configuration.shuffle, pin_memory=True,
                                     num_workers=configuration.num_workers, drop_last=True)

    validation_dataloader = DataLoader(validation_dataset, configuration.test_batch_size, pin_memory=True,
                                       num_workers=configuration.num_workers, drop_last=True)

    return training_dataloader, validation_dataloader
