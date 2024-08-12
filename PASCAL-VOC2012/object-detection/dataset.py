import cv2
import numpy as np
from torchvision.datasets import VOCSegmentation
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class PascalVOCDataset(VOCSegmentation):
    VOC_CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    VOC_COLORMAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
    
    def __init__(self, 
                 root="~/data/pascal_voc", 
                 image_set="train", 
                 download=True, 
                 transform=A.Compose([
                     A.Resize(height=160, width=256), 
                     A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255), 
                     ToTensorV2(transpose_mask=True)])
    ):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    @classmethod
    def convert_color_mask_to_class_onehot(cls, color):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = color.shape[:2]
        target = np.zeros((height, width, len(cls.VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(cls.VOC_COLORMAP):
            target[:, :, label_index] = np.all(color==label, axis=-1).astype(float)
        return target
    
    @classmethod
    def convert_class_to_color_mask(cls, segmentation, segmentation_axis):
        segmentation = np.asarray(segmentation)
        segmentation = np.argmax(segmentation, axis=segmentation_axis)
        image = np.zeros((*segmentation.shape, 3))
        for i, color in enumerate(cls.VOC_COLORMAP):
            image[segmentation==i, :] = np.asarray(color)
        image = image.astype(np.uint8)
        return image

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert_color_mask_to_class_onehot(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask