import numpy as np
import os 
import cv2 as cv
from skimage.exposure import rescale_intensity
import SimpleITK as sitk

def specific_intensity_window_1(image, window_percent=0.1):
    image = image.astype('int64') 
    arr = np.asarray_chkfinite(image)
    min_val = arr.min()
    number_of_bins = arr.max() - min_val + 1
    hist = np.bincount((arr-min_val).ravel(), minlength=number_of_bins)
    hist_new = hist[1:]
    total = np.sum(hist_new)
    window_low = window_percent * total
    window_high = (1 - window_percent) * total
    cdf = np.cumsum(hist_new)
    low_intense = np.where(cdf >= window_low) + min_val
    high_intense = np.where(cdf >= window_high) + min_val
    res = rescale_intensity(image, in_range=(low_intense[0][0], high_intense[0][0]),out_range=(arr.min(), arr.max()))
    return res

def dice(im1, im2):
   
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def contour_size(contour):
    return contour.shape[0]

def apply_mask(image, mask):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if mask[x,y]==0:
                image[x,y]=0
    return image

def fill_holes(label):
    '''
    It fills the holes (for example, for abdominal aorta, it fills the lumen hole that appears in the mask)
    :param label: label where the holes will be removed (simple itk)
    :return: label with removed holes
    '''

    fillHole=sitk.GrayscaleFillholeImageFilter

    if label.GetDimension()==3:
        x, y, z= label.GetSize()
        np_label = sitk.GetArrayFromImage(label)

        for slice in range(z):

            #para asignar un conjunto de filas y columnas hay que utilizar numpy
            noHole = sitk.GrayscaleFillhole(label[:, :, slice])

            np_label[slice, :, :] = sitk.GetArrayFromImage(noHole)

    elif label.GetDimension()==2:
        noHole = sitk.GrayscaleFillhole(label)
        np_label = sitk.GetArrayFromImage(noHole)

    withoutHoles=sitk.GetImageFromArray(np_label)
    withoutHoles.SetOrigin(label.GetOrigin())
    withoutHoles.SetSpacing(label.GetSpacing())

    return withoutHoles

def blur_half_img(image_series):
    n = image_series.shape[2]
    output_series = np.zeros((n, 96, 96, 2), dtype=image_series.dtype)
    for i in range(n):
        image = image_series[:,:,i]
        right_img = image.copy()
        left_img = image.copy()
        left_img[24:48, 55:] = image[-24:, :41]
        left_img[48:72, 55:] = image[:24, :41]
        res_1 = cv.GaussianBlur(left_img, (5, 5), 3)
        left_img[:, 48:] = res_1[:, 48:]
        right_img[24:48, :41] = image[-24:, 55:]
        right_img[48:72, :41] = image[:24, 55:]
        res_2 = cv.GaussianBlur(right_img, (5, 5), 3)
        right_img[:, :48] = res_2[:, :48]
        output_series[i, :, :, 0] = left_img
        output_series[i, :, :, 1] = right_img
    return output_series

def resize_image(image, new_size):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [(original_spacing[i] * original_size[i] / new_size[i]) for i in range(len(new_size))]
    resampled_image = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkLinear,
                                    image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                                    image.GetPixelID())
    return resampled_image

import cv2 as cv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt



def orientation_detection(img): 
    # Apply a Gaussian blur to the image to remove noise
    gray = cv.GaussianBlur((img*255).astype(np.uint8), (5, 5), 0)
    # Apply Canny edge detection to detect edges in the image
    edges = cv.Canny((gray*255).astype(np.uint8), 50, 100)

    # Find contours in the image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)

    # Iterate through each contour and determine whether it corresponds to a right or left mask
    for contour in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv.boundingRect(contour)
        
        # Determine the center of the bounding box
        cx = x + w // 2
        cy = y + h // 2
        
        # If the center of the bounding box is on the left side of the image, it is a left mask; otherwise, it is a right mask
        # Right
        if cx < img.shape[1] // 2:
            mask[y:y+h, x:x+w] = 1
        # Left
        else:
            mask[y:y+h, x:x+w] = 2
    return mask



def label_right_left(img): 
    '''
    It receives a sitk format image and returns an array with relabeled mask. Right mask = label 1. Left mask = label 2. 
    '''
    mask_right_total = np.zeros_like(img)
    mask_left_total = np.zeros_like(img)

    if len(img.shape) == 2: # 2d image
        img_unique = img
        mask = orientation_detection(img_unique)

        # Relabel original mask
        mask_right = np.logical_and(img_unique > 0, mask == 1)
        mask_left = np.logical_and(img_unique > 0, mask == 2)

        mask_right_total = mask_right
        mask_left_total = mask_left


    else:
        for i in range(0, img.shape[0]): 
            img_unique= img[i,:,:]
            mask = orientation_detection(img_unique)

            # Relabel original mask
            mask_right = np.logical_and(img_unique > 0, mask == 1)
            mask_left = np.logical_and(img_unique > 0, mask == 2)

            # mask_total[i, :, :] = mask_dual
            mask_right_total[i:] = mask_right
            mask_left_total[i:] = mask_left

    return mask_right_total.astype(np.uint8), mask_left_total.astype(np.uint8)


