import numpy as np
from dataset import PascalVOCDataset

def segmentation_to_image(segmentation, segmentation_axis):
    segmentation = np.asarray(segmentation)
    segmentation = np.argmax(segmentation, axis=segmentation_axis)
    image = np.zeros((*segmentation.shape, 3))
    for i, color in enumerate(PascalVOCDataset.VOC_COLORMAP):
        image[segmentation==i, :] = np.asarray(color)
    image = image.astype(np.uint8)
    return image