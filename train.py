import neptune
import os
import sys
import warnings
from tokenize import Imagnumber
from cv2 import rotate
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tensorflow as tf
import imgaug
import skimage.color
import skimage.io
import skimage.transform
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from mrcnn.config import Config
from scipy.ndimage import binary_fill_holes
from mrcnn import utils
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn import visualize
from preprocessing import specific_intensity_window_1
import preprocessing 

if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print("GPU is available")
else:
    print("GPU is not available")

def get_ax(rows=1, cols=1, size=7):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

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

class KidneyConfig(Config):
    NAME = 'kidney'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    NUM_CLASSES = 1 + 1 # KIDNEY, FOREGROUND
    DETECTION_MIN_CONFIDENCE = 0.85
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256
    BACKBONE='resnet50'
    IMAGE_SHAPE = [256, 256, 3]
    STEPS_PER_EPOCH = 150
    DETECTION_MAX_INSTANCES = 10
    LEARNING_RATE = 0.0002
    USE_MINI_MASK = False
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_SCALES = [32, 64, 128, 256, 512]
    MAX_GT_INSTANCES = 2 
    IMAGE_CHANNEL_COUNT = 3

class KidneyDataset(utils.Dataset):
    def load_kidney_scan(self, dataset_dir, subset):
        # Adding classes for cortex and medulla
        self.add_class("kidney", 1, "cortex")
        self.add_class("kidney", 2, "medulla")
        
        # Verify subset is correct
        assert subset in ["train", "val", "test"]
        
        # Path to images
        dataset_dir = os.path.join(dataset_dir, subset, 'Images')
        file_names = os.listdir(dataset_dir)
        
        for image_id in file_names:
            image_path = os.path.join(dataset_dir, image_id)
            
            # Read image using SimpleITK and process it
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            height, width = image.shape[:2]
            
            # Example: Rescale and preprocess the image if needed
            image_sitk = sitk.GetImageFromArray(image)
            image_rescaled = sitk.RescaleIntensity(image_sitk, 0, 255)
            # image_rescaled = specific_intensity_window_1(sitk.GetArrayFromImage(image_rescaled), window_percent=0.15)
            
            # Update image (optional)
            image = image_rescaled
            
            # Add the image to the dataset
            self.add_image(
                "kidney",
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                subset=subset
            )
    def load_mask(self, image_id):
        """Load and return the mask for cortex and medulla segmentation.

        Args:
        - image_id: The ID of the image to load the mask for.

        Returns:
        - mask: A binary array of shape [height, width, 2] where each channel represents a class.
        - class_ids: A 1D array of shape [2] with class IDs.
        """
        # Retrieve the image info for the current image_id
        image_info = self.image_info[image_id]
        height, width = image_info["height"], image_info["width"]

        # Define the subset (train, val, test)
        subset = image_info["subset"]

        # Define paths to the cortex and medulla masks
        cortex_mask_path = os.path.join(DATASET_DIR, subset, 'masks/Cortex', image_info['id'])
        medulla_mask_path = os.path.join(DATASET_DIR, subset, 'masks/Medulla', image_info['id'])

        # Initialize a list for holding individual masks and class IDs
        masks = []
        class_ids = []

        # Read and process the cortex mask if it exists
        if os.path.exists(cortex_mask_path):
            cortex_mask = sitk.GetArrayFromImage(sitk.ReadImage(cortex_mask_path))
            cortex_mask = cortex_mask[0,:,:] # las cortezas tienen shape = (1,96,96)
            cortex_mask = (cortex_mask > 0).astype(np.uint8)  # Ensure binary mask
            masks.append(cortex_mask)  # Append to mask list
            class_ids.append(1)  # Cortex class ID

        # Read and process the medulla mask if it exists
        if os.path.exists(medulla_mask_path):
            medulla_mask = sitk.GetArrayFromImage(sitk.ReadImage(medulla_mask_path))
            medulla_mask = medulla_mask[0,:,:] # las medulas tienen shape = (1,96,96)
            medulla_mask = (medulla_mask > 0).astype(np.uint8)  # Ensure binary mask
            masks.append(medulla_mask)  # Append to mask list
            class_ids.append(2)  # Medulla class ID

        # If no masks are found, return an empty mask with an empty class ID
        if len(masks) == 0:
            mask = np.zeros((height, width, 0), dtype=np.uint8)
            class_ids = np.array([])
        else:
            # Stack all the masks along the third dimension
            mask = np.stack(masks, axis=-1)

            # Check if the mask has the shape (height, width, 2)
            # If not, add a zero-filled mask for the missing class
            if mask.shape[-1] == 1:  # Only one class exists
                # Check which class is missing (either cortex or medulla)
                if class_ids[0] == 1:
                    # Cortex is present; add an empty medulla mask
                    medulla_mask = np.zeros((height, width), dtype=np.uint8)
                    mask = np.stack([mask[..., 0], medulla_mask], axis=-1)
                    class_ids = np.array([1, 2], dtype=np.int32)
                else:
                    # Medulla is present; add an empty cortex mask
                    cortex_mask = np.zeros((height, width), dtype=np.uint8)
                    mask = np.stack([cortex_mask, mask[..., 0]], axis=-1)
                    class_ids = np.array([1, 2], dtype=np.int32)

        # Verify the mask has the expected shape
        assert mask.shape == (height, width, 2), \
            f"Expected mask shape {(height, width, 2)}, but got {mask.shape}"

        # Print debug information
        print(f"Mask shape: {mask.shape}")
        print(f"Class IDs: {class_ids}")
        print(f"Cortex mask sum: {np.sum(mask[..., 0])}")
        print(f"Medulla mask sum: {np.sum(mask[..., 1])}")

        # Return the mask and associated class IDs
        return mask, class_ids
    

class NeptuneCallback(tf.keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, value in logs.items():
            self.run[f"metrics/{metric_name}"].log(value)

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


def prepare_synthetic_2kidneys(img_path, cortex_mask_path, medulla_mask_path, out_path): 
    # Create folders if they do not exist
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_path, folder), exist_ok=True)
        os.makedirs(os.path.join(out_path, folder, 'masks/Cortex'), exist_ok=True)
        os.makedirs(os.path.join(out_path, folder, 'masks/Medulla'), exist_ok=True)

    for model in models: 
        for nimg in range(2, 52): 
            for nslice in range(1, 4): 
                filename = f'SpinEcho_Model_{model}_aslSlice_{nslice}_nIm_{nimg}.nii'
                img = sitk.ReadImage(os.path.join(img_path, filename))
                resized_image = resize_image(img, (1, 96, 96))
                rescaled_image = sitk.RescaleIntensity(resized_image, 0, 255)
                blurred_img = blur_half_img(sitk.GetArrayFromImage(rescaled_image))
                right_img = blurred_img[0, :, :, 0]
                left_img = blurred_img[0, :, :, 1]

                # Load masks from the respective paths
                cortex_mask_file = os.path.join(cortex_mask_path, filename)
                medulla_mask_file = os.path.join(medulla_mask_path, filename)

                cortex_mask = sitk.ReadImage(cortex_mask_file) if os.path.exists(cortex_mask_file) else None
                medulla_mask = sitk.ReadImage(medulla_mask_file) if os.path.exists(medulla_mask_file) else None

                cortex_mask_right, cortex_mask_left = preprocessing.label_right_left(sitk.GetArrayFromImage(cortex_mask))
                medulla_mask_right, medulla_mask_left = preprocessing.label_right_left(sitk.GetArrayFromImage(medulla_mask))
                
                # Create masks for cortex and medulla
                right_mask_cortex = np.zeros((1, 96, 96), dtype=np.uint8)
                left_mask_cortex = np.zeros((1, 96, 96), dtype=np.uint8)
                right_mask_medulla = np.zeros((1, 96, 96), dtype=np.uint8)
                left_mask_medulla = np.zeros((1, 96, 96), dtype=np.uint8)

                if cortex_mask is not None:
                    right_mask_cortex[cortex_mask_right == 1] = 1  # Assign class '1' for cortex
                    left_mask_cortex[cortex_mask_left == 1] = 1

                if medulla_mask is not None:
                    right_mask_medulla[medulla_mask_right == 1] = 1  # Assign class '1' for medulla
                    left_mask_medulla[medulla_mask_left == 1] = 1

                right_filename = f'SpinEcho_Model_{model}_nIm_{nimg}_Right_Slice_{nslice}.nii'
                left_filename = f'SpinEcho_Model_{model}_nIm_{nimg}_Left_Slice_{nslice}.nii'

                # Determine destination folder for images
                if model in train_models:
                    dest_folder_img = 'train/images/'
                elif model in val_models:
                    dest_folder_img = 'val/images/'
                elif model in test_models:
                    dest_folder_img = 'test/images/'
                else:
                    continue  # If the model is not in any list, skip

                # Construct full destination paths for images
                right_file_path_img = os.path.join(out_path, dest_folder_img, right_filename)
                left_file_path_img = os.path.join(out_path, dest_folder_img, left_filename)

                # Remove existing files if they exist
                if os.path.exists(right_file_path_img):
                    os.remove(right_file_path_img)
                if os.path.exists(left_file_path_img):
                    os.remove(left_file_path_img)

                # Save images
                right_img_sitk = sitk.GetImageFromArray(right_img)
                left_img_sitk = sitk.GetImageFromArray(left_img)
                sitk.WriteImage(right_img_sitk, right_file_path_img)
                sitk.WriteImage(left_img_sitk, left_file_path_img)

                # Construct full destination paths for masks
                right_file_path_mask_cortex = os.path.join(out_path, dest_folder_img.replace('images', 'masks/Cortex'), right_filename)
                left_file_path_mask_cortex = os.path.join(out_path, dest_folder_img.replace('images', 'masks/Cortex'), left_filename)
                right_file_path_mask_medulla = os.path.join(out_path, dest_folder_img.replace('images', 'masks/Medulla'), right_filename)
                left_file_path_mask_medulla = os.path.join(out_path, dest_folder_img.replace('images', 'masks/Medulla'), left_filename)

                # Remove existing mask files if they exist
                if os.path.exists(right_file_path_mask_cortex):
                    os.remove(right_file_path_mask_cortex)
                if os.path.exists(left_file_path_mask_cortex):
                    os.remove(left_file_path_mask_cortex)
                if os.path.exists(right_file_path_mask_medulla):
                    os.remove(right_file_path_mask_medulla)
                if os.path.exists(left_file_path_mask_medulla):
                    os.remove(left_file_path_mask_medulla)

                # Save cortex masks
                sitk.WriteImage(sitk.GetImageFromArray(right_mask_cortex), right_file_path_mask_cortex)
                sitk.WriteImage(sitk.GetImageFromArray(left_mask_cortex), left_file_path_mask_cortex)

                # Save medulla masks
                sitk.WriteImage(sitk.GetImageFromArray(right_mask_medulla), right_file_path_mask_medulla)
                sitk.WriteImage(sitk.GetImageFromArray(left_mask_medulla), left_file_path_mask_medulla)


def copy_patient_files(patient_id, source_dir, target_dir):
    source_images_dir = os.path.join(source_dir, 'images')
    source_masks_dir = os.path.join(source_dir, 'masks')
    target_images_dir = os.path.join(target_dir, 'images')
    target_masks_dir = os.path.join(target_dir, 'masks')

    # Copy image files
    for file_name in os.listdir(source_images_dir):
        if patient_id in file_name:
            shutil.copy(os.path.join(source_images_dir, file_name), os.path.join(target_images_dir, file_name))

    # Copy mask files
    for file_name in os.listdir(source_masks_dir):
        if patient_id in file_name:
            shutil.copy(os.path.join(source_masks_dir, file_name), os.path.join(target_masks_dir, file_name))

def remove_files_containing_keyword(root_dir, keyword):
    for subdir, _, _ in os.walk(root_dir):
        # Use glob to find all files containing the keyword
        files = glob.glob(os.path.join(subdir, f'*{keyword}*'))
        for file in files:
            os.remove(file)
            print(f"Deleted file: {file}")


if __name__ == '__main__':
    
    ####################################################
    ########### GENERAL SETTINGS #######################
    ####################################################

    training_data_path = ''
    
    # Create directories
    train_path = os.path.join(training_data_path, 'train')
    val_path = os.path.join(training_data_path, 'val')
    test_path = os.path.join(training_data_path, 'test')
    train_images_dir = os.path.join(train_path, 'images')
    train_masks_dir = os.path.join(train_path, 'masks')
    val_images_dir = os.path.join(val_path, 'images')
    val_masks_dir = os.path.join(val_path, 'masks')
    test_images_dir = os.path.join(test_path, 'images')
    test_masks_dir = os.path.join(test_path, 'masks')

        
    ####################################################
    ########### TRAINING######## #######################
    ####################################################
    
    run = neptune.init_run(
        project='aod7/Kidney-Seg-Synthetic',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwM2Q3YjZlZi03NjUzLTRlMWUtYmQ5Mi1kYzA2NDMzZjFhOGQifQ=='
    )
    DATASET_DIR = 'Z:/RM_RENAL/Data_Segmentation/'
    DEFAULT_LOGS_DIR = os.path.join(DATASET_DIR, 'logs')
    WEIGHTS_PATH = 'Z:/weights/mask_rcnn_coco.h5'

    ROOT_DIR = os.path.join('Mask_RCNN-master')
    sys.path.append(ROOT_DIR)

    config = KidneyConfig()

    augmentation = imgaug.augmenters.Sometimes(1, imgaug.augmenters.OneOf([
        imgaug.augmenters.Affine(rotate=(-4,4)),
        imgaug.augmenters.Affine(rotate=(-6,6)),
        imgaug.augmenters.Affine(rotate=(-8,8)),
        imgaug.augmenters.Affine(rotate=(-10,10)),
        imgaug.augmenters.Affine(translate_px={"x": (-10, 10)}),
        imgaug.augmenters.Affine(translate_px={ "y": (-10, 10)}),
        imgaug.augmenters.Fliplr(1),
        imgaug.augmenters.Flipud(1)
    ]))

    dataset_train = KidneyDataset()
    dataset_train.load_kidney_scan(DATASET_DIR, 'train')
    dataset_train.prepare()

    dataset_val = KidneyDataset()
    dataset_val.load_kidney_scan(DATASET_DIR, 'val')
    dataset_val.prepare()

    dataset_test = KidneyDataset()
    dataset_test.load_kidney_scan(DATASET_DIR, 'test')
    dataset_test.prepare()

    model = modellib.MaskRCNN(
        mode='training',
        config=config,
        model_dir=DEFAULT_LOGS_DIR
    )

    print('Loading coco weights...')
    model.load_weights(
        WEIGHTS_PATH, # cambiar esto a KIDNEY_MODEL_PATH (estoy entrnando con los pesos de izaskun)
        by_name=True, 
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    neptune_callback = NeptuneCallback(run)

    print('Training started....')
    # Unfreeze all layers in the model
    for layer in model.keras_model.layers:
        layer.trainable = True

    # Fine-tune the entire model
    model.train(
        dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,  # Use a lower learning rate for fine-tuning
        epochs=150,  # Continue training for more epochs
        layers='all',  # Train all layers
        augmentation=augmentation,
        custom_callbacks=[neptune_callback]
    )

    print('Training finished.')
    run.stop()