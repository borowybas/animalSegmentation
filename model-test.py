import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch
import warnings
import albumentations as albu
from collections import namedtuple
from PIL import Image
from torch.backends import mps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

warnings.filterwarnings("ignore")
# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
ENCODER = 'tu-ssl_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [cls.color for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        if isinstance(i, slice):
            start = 0 if not i.start else i.start
            stop = len(self) if not i.stop else i.stop
            step = 1 if not i.step else i.step
            return [self.__getitem__(j) for j in range(start, stop, step)]

        if not os.path.exists(self.images_fps[i]):
            raise FileNotFoundError(self.images_fps[i])

        if not os.path.exists(self.masks_fps[i]):
            raise FileNotFoundError(self.masks_fps[i])

        # read data
        image = np.array(Image.open(self.images_fps[i]).convert("RGB"))
        mask = np.array(Image.open(self.masks_fps[i]).convert("RGB"))

        # resize images
        image = np.array(Image.fromarray(image).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))

        # extract certain classes from mask (e.g. cars)

        masks = [(mask == v) for v in self.class_values]
        masks = [np.all(m, axis=2) for m in masks]

        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # image = np.moveaxis(image, -1, 0)
        # mask = np.moveaxis(mask, -1, 0)
        # mask = np.expand_dims(mask, 0)

        print("Image path: ", self.ids[i])
        print(image.shape, mask.shape)

        return image, mask

    def __len__(self):
        return len(self.ids)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


Class = namedtuple("Class", ['name', 'color'])

CLASSES = [Class('NonMaskingBackground', np.array([255, 0, 0])),
           Class('MaskingBackground', np.array([0, 255, 0])),
           Class('Animal', np.array([0,0,255])),
           Class('NonMaskingForegroundAttention', np.array([255, 255, 255]))]

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]


T_DATA_DIR = './dataset/test/'

# load repo with data if it is not exists
if not os.path.exists(T_DATA_DIR):
    raise FileNotFoundError(f"Directory {T_DATA_DIR} does not exist.")

t_images_dir = os.path.join(T_DATA_DIR, 'Images')
t_masks_dir = os.path.join(T_DATA_DIR, 'Masks')

# load best saved checkpoint
best_model = torch.load('./best_model.pth')

if __name__ == 'model-test':
    test_dataset = Dataset(t_images_dir, t_masks_dir,
                          classes=CLASSES[2:3],
                          preprocessing=get_preprocessing(preprocessing_fn))
    batch_size = 10
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE
    )

    logs = test_epoch.run(test_dataloader)

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        t_images_dir, t_masks_dir,
        classes=CLASSES[2:3],
    )

    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )