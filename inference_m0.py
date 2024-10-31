import os
import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import tensorflow as tf
import time
import pandas as pd
import skimage.color
from mrcnn import utils
from preprocessing import specific_intensity_window_1
from mrcnn.config import Config
import preprocessing

class KidneyDataset(utils.Dataset):
    def load_kidney_scan(self, dataset_dir, subset):
        self.add_class("tumor", 1, "tumor")
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset, 'Images')
        file_names = os.listdir(dataset_dir)
        for image_id in file_names:
            image_path = os.path.join(dataset_dir, image_id)
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            height, width = image.shape[:2]
            image = specific_intensity_window_1(image, window_percent=0.15)
            self.add_image(
                "tumor",
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                subset = subset
            )

    def load_mask(self, image_id):
        # Retrieve the information for the current image
        image_info = self.image_info[image_id]

        # Define the mask shape based on image width
        mask_shape = (image_info["width"], image_info["width"], 3)
        
        # Create an empty mask with the desired shape
        mask = np.zeros(mask_shape, dtype=np.uint8)

        # Determine the subset and construct the mask path
        subset = image_info["subset"]
        path = os.path.join(DATASET_DIR, subset, 'masks')
        files_names = os.listdir(path)
        mask_path = os.path.join(path, files_names[image_id])

        # Read the mask image using SimpleITK
        sitk_mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(sitk_mask)

        # Ensure the mask array matches the desired shape
        # Assuming the mask_array is 2D, we need to expand it to 3D
        if mask_array.ndim == 2:
            mask_array = np.stack([mask_array] * 3, axis=-1)

        # Resize the mask_array to match mask_shape
        mask[:,:,:] = mask_array[:mask_shape[0], :mask_shape[1], :mask_shape[2]]
        # mask_reshaped = np.transpose(mask, (2, 1, 0))
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "kidney":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class KidneyConfig(Config):
    NAME = 'kidney'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.85
    IMAGE_MAX_DIM = 128
    IMAGE_MIN_DIM = 128
    BACKBONE='resnet50'
    IMAGE_SHAPE = [128, 128, 3]
    STEPS_PER_EPOCH = 100
    DETECTION_MAX_INSTANCES = 2
    LEARNING_RATE = 0.0002
    USE_MINI_MASK = False
    RPN_ANCHOR_RATIOS = [0.5,1,2]

class InferenceConfig(KidneyConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0

def calculate_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    dice = 2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice

def evaluate(model, dataset, output_masks_folder):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = dataset.image_ids

    t_prediction = 0
    t_start = time.time()

    dice_scores = []
    results_df = pd.DataFrame(columns=['ImageName', 'nSlice', 'DiceScore', 'PredictionTime'])
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        #image_rescaled = specific_intensity_window_1(image, window_percent=0.15)
        gt_mask = dataset.load_mask(image_id)[0]

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        filename = os.path.splitext(dataset.image_info[image_id]['id'])[0]
        # print(filename)  
  
        # Mayor máscara = kidney
        try:
            volume1 = r["masks"][:,:,0].astype(bool)
            dice1 = calculate_dice(gt_mask[:,:,0], volume1)
            try:
                volume2 = r["masks"][:,:,1].astype(bool)
                dice2 = calculate_dice(gt_mask[:,:,0], volume2)

                # if np.sum(volume1) >= np.sum(volume2): 
                if dice1 >= dice2:
                    pred_mask = volume1
                    dice = dice1
                elif dice2 > dice1:
                    pred_mask = volume2
                    dice = dice2
            except: 
                pred_mask = volume1
                dice = dice1
        except:
            pass
                
        # dice = calculate_dice(gt_mask[:,:,0], pred_mask)
        dice_scores.append(dice)

        # Save results to DataFrame
        results_df = results_df.append({
                                        'ImageName': filename,
                                        'PredictionTime': t_prediction,
                                        'DiceScore': dice,
                                        }, ignore_index=True)


        mask_filename = f"{filename}.nii"
        mask_filepath = os.path.join(output_masks_folder + 'seg/', mask_filename)
        sitk.WriteImage(sitk.GetImageFromArray(pred_mask.astype(np.uint8)), mask_filepath)

    return results_df

def evaluate_tk(model, data_path, studies, predictions_path):
    t_prediction = 0
    slice_results_df = pd.DataFrame(columns=['Study', 'SliceNumber', 'DiceScore'])

    for i, nstudies in enumerate(studies):
        filename = nstudies.replace('/', '_')
        print('Evaluating study: ', filename)

        # Load image paths
        images_path = data_path + 'images/' + nstudies
        masks_path = data_path + 'masks/' + nstudies

        for nslice in range(0, 3):
            pred_sum = None
            gt_sum = None
            images = sitk.ReadImage(images_path + '1.nii')
            gt_masks = sitk.ReadImage(masks_path + '1.mha')
            gt_masks = preprocessing.fill_holes(gt_masks)

            image = images[:, :, nslice]
            gt_mask = gt_masks[:, :, nslice]
            image = preprocess_img(image, rescale=True, int_window=False)
            gt_mask = preprocess_img(gt_mask)

            # Run detection
            t = time.time()
            r = model.detect([image], verbose=0)[0]
            t_prediction += (time.time() - t)

            # Determine the largest mask as kidney
            try:
                volume1 = r["masks"][:, :, 0].astype(bool)
                dice1 = calculate_dice(gt_mask[:, :, 0], volume1)
                try:
                    volume2 = r["masks"][:, :, 1].astype(bool)
                    dice2 = calculate_dice(gt_mask[:, :, 0], volume2)
                    if dice1 >= dice2:
                        pred_mask = volume1
                        dice = dice1
                    else:
                        pred_mask = volume2
                        dice = dice2
                except:
                    pred_mask = volume1
                    dice = dice1
            except:
                print('No detection is made')
                pred_mask = np.zeros_like(image[:, :, 0])  # Si no se detecta nada, usar una máscara vacía
                dice = 0

            # Save slice-level results
            slice_results_df = slice_results_df.append({
                'Study': filename.split('_')[0],
                'SliceNumber': nslice + 1,
                'DiceScore': dice
            }, ignore_index=True)

            mask_filename = f'{filename}Slice_{nslice + 1}_nIm_1.nii'
            os.makedirs(predictions_path + 'seg/', exist_ok=True)
            mask_filepath = os.path.join(predictions_path + 'seg/', mask_filename)
            pred_mask = pred_mask.astype('uint8')
            resized_pred = resize_image(sitk.GetImageFromArray(pred_mask), (96,96), is_mask=True)
            sitk.WriteImage(resized_pred, mask_filepath)

    return slice_results_df

def evaluate_shk(model, images_path, masks_path, studies, predictions_path):
    '''
    Las imágenes están divididas en Right y Left. 306 máscaras. 
    '''
    t_prediction = 0
    results_df = pd.DataFrame(columns=['Filename', 'RightDiceScore', 'LeftDiceScore'])
    
    for nstudies in studies:
        print('Evaluating study: ', nstudies)        
        
        for nimgs in range(2,52): 
            
            for nslice in range(1,4): 
                image = sitk.ReadImage(images_path + f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_{nimgs}.nii')
                image_rescaled = sitk.RescaleIntensity(image, 0, 255)
                image_resized = resize_image(image_rescaled, (3, 96, 96), is_mask=False)

                gt_mask = sitk.ReadImage(masks_path + f'SpinEcho_Model_{nstudies}_nIm_{nimgs}_Slice_{nslice}.nii')
                gt_mask = preprocess_img(gt_mask)

                # Run detection
                t = time.time()
                r = model.detect([image_resized], verbose=0)[0]
                t_prediction += (time.time() - t)

                # For Right detection
                try:
                    volume1 = r["masks"][:,:,0].astype(bool)
                    dice1 = calculate_dice(gt_mask[:,:,0], volume1)
                    try:
                        volume2 = r["masks"][:,:,1].astype(bool)
                        dice2 = calculate_dice(gt_mask[:,:,0], volume2)
                        if dice1 >= dice2:
                            pred_mask = volume1
                            dice = dice1
                        elif dice2 > dice1:
                            pred_mask = volume2
                            dice = dice2
                    except: 
                        pred_mask = volume1
                        dice = dice1
                except:
                    print('No detection is made')
                    dice = 0


                # Store the results in the DataFrame
                filename = f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_{nimgs}'
                results_df = results_df.append({'Filename': filename, 
                                                'DiceScore': dice, 
                                                }, ignore_index=True)
                
                # pred_mask.astype('uint8')
                # mask_rgb = skimage.color.gray2rgb(pred_mask)
                # mask_filepath = os.path.join(predictions_path + 'seg/', filename + '.nii')
                # os.makedirs(predictions_path + 'seg/', exist_ok=True)
                # sitk.WriteImage(sitk.GetImageFromArray(mask_rgb), mask_filepath)

    return results_df

def evaluate_stk(model, images_path, masks_path, studies, predictions_path):
    '''
    Las imágenes están divididas en Right y Left. 306 máscaras. 
    '''
    t_prediction = 0
    results_df = pd.DataFrame(columns=['Filename', 'RightDiceScore', 'LeftDiceScore'])
    dice_accumulated_df = pd.DataFrame(columns=['Study', 'DiceScoreRightAccumulated', 'DiceScoreLeftAccumulated'])

    for nstudies in studies:
        print('Evaluating study: ', nstudies)  

        pred_sum_right = None
        pred_sum_left = None
        gt_sum_right = None
        gt_sum_left = None
        
        for nslice in range(1, 4): 
            image = sitk.ReadImage(images_path + f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_1.nii')
            image_rescaled = sitk.RescaleIntensity(image, 0, 255)
            image_resized = resize_image(image_rescaled, (3, 96, 96), is_mask=False)
            image_blurred = preprocessing.blur_half_img(sitk.GetArrayFromImage(image_resized))
            right_img = image_blurred[0, :, :, 0]
            right_img = preprocess_img(sitk.GetImageFromArray(right_img), rescale=False, int_window=False)
            left_img = image_blurred[0, :, :, 1]
            left_img = preprocess_img(sitk.GetImageFromArray(left_img), rescale=False, int_window=False)

            gt_right_mask = sitk.ReadImage(masks_path + f'SpinEcho_Model_{nstudies}_nIm_1_Right_Slice_{nslice}.nii')
            gt_right_mask = preprocess_img(gt_right_mask)
            gt_left_mask = sitk.ReadImage(masks_path + f'SpinEcho_Model_{nstudies}_nIm_1_Left_Slice_{nslice}.nii')
            gt_left_mask = preprocess_img(gt_left_mask)

            # Run detection
            t = time.time()
            r_right = model.detect([right_img], verbose=0)[0]
            r_left = model.detect([left_img], verbose=0)[0]
            t_prediction += (time.time() - t)

            # For Right detection
            try:
                right_volume1 = r_right["masks"][:,:,0].astype(bool)
                right_dice1 = calculate_dice(gt_right_mask[:,:,0], right_volume1)
                try:
                    right_volume2 = r_right["masks"][:,:,1].astype(bool)
                    right_dice2 = calculate_dice(gt_right_mask[:,:,0], right_volume2)
                    if right_dice1 >= right_dice2:
                        right_pred_mask = right_volume1
                        right_dice = right_dice1
                    else:
                        right_pred_mask = right_volume2
                        right_dice = right_dice2
                except: 
                    right_pred_mask = right_volume1
                    right_dice = right_dice1
            except:
                print('No detection is made')
                right_pred_mask = np.zeros_like(right_img[:,:,0])
                right_dice = 0

            # For Left detection
            try:
                left_volume1 = r_left["masks"][:,:,0].astype(bool)
                left_dice1 = calculate_dice(gt_left_mask[:,:,0], left_volume1)
                try:
                    left_volume2 = r_left["masks"][:,:,1].astype(bool)
                    left_dice2 = calculate_dice(gt_left_mask[:,:,0], left_volume2)
                    if left_dice1 >= left_dice2:
                        left_pred_mask = left_volume1
                        left_dice = left_dice1
                    else:
                        left_pred_mask = left_volume2
                        left_dice = left_dice2
                except: 
                    left_pred_mask = left_volume1
                    left_dice = left_dice1
            except:
                print('No detection is made')
                left_pred_mask = np.zeros_like(left_img[:,:,0])
                left_dice = 0

            # Acumular las máscaras predichas y de verdad terreno
            if pred_sum_right is None:
                pred_sum_right = right_pred_mask.astype(np.int16)
                gt_sum_right = gt_right_mask[:,:,0].astype(np.int16)
            else:
                pred_sum_right += right_pred_mask.astype(np.int16)
                gt_sum_right += gt_right_mask[:,:,0].astype(np.int16)

            if pred_sum_left is None:
                pred_sum_left = left_pred_mask.astype(np.int16)
                gt_sum_left = gt_left_mask[:,:,0].astype(np.int16)
            else:
                pred_sum_left += left_pred_mask.astype(np.int16)
                gt_sum_left += gt_left_mask[:,:,0].astype(np.int16)

            # Store the results in the DataFrame
            filename = f'SpinEcho_Model_{nstudies}_aslSlice_{nslice}_nIm_1'
            results_df = results_df.append({'Filename': filename, 
                                            'RightDiceScore': right_dice, 
                                            'LeftDiceScore': left_dice}, ignore_index=True)
            
            total_mask = right_pred_mask.astype('uint8') + left_pred_mask.astype('uint8')
            total_mask_resized = resize_image(sitk.GetImageFromArray(total_mask), (96, 96), is_mask=True)
            mask_filepath = os.path.join(predictions_path + 'seg/', filename + '.nii')
            os.makedirs(predictions_path + 'seg/', exist_ok=True)
            sitk.WriteImage(total_mask_resized, mask_filepath)

    return results_df

def evaluate_hk(model, img_path, mask_path, studies, predictions_path):

    results_df = pd.DataFrame(columns=['Filename', 'RightDiceScore', 'LeftDiceScore'])

    for i, nstudies in enumerate(studies):
        filename = nstudies.replace('/', '_')
        print('Evaluating study: ', filename)

        # Load image paths
        images_path = img_path + nstudies + 'imgs_to_register.mhd'
        masks_path = mask_path + nstudies + 'kidney_2d.nii'

        images = sitk.ReadImage(images_path)
        gt_mask = sitk.ReadImage(masks_path)[:, :, 0]
        gt_mask = resize_image(gt_mask, (96, 96), is_mask=True)
        
        image = images[:, :, 0]
        image = preprocess_img(image, rescale=True, int_window=False)
        image_resized = resize_image(sitk.GetImageFromArray(image), (3, 96, 96), is_mask=False)
        
        image_blurred = preprocessing.blur_half_img(sitk.GetArrayFromImage(image_resized))
        right_img = image_blurred[0, :, :, 0]
        right_img = preprocess_img(sitk.GetImageFromArray(right_img), rescale=True, int_window=False)
        left_img = image_blurred[0, :, :, 1]
        left_img = preprocess_img(sitk.GetImageFromArray(left_img), rescale=True, int_window=False)
        left_img.astype('int16')

        gt_right_mask, gt_left_mask = preprocessing.label_right_left(sitk.GetArrayFromImage(gt_mask))
        gt_right_mask = preprocess_img(sitk.GetImageFromArray(gt_right_mask))
        gt_left_mask = preprocess_img(sitk.GetImageFromArray(gt_left_mask))

        # Run detection
        t = time.time()
        r_right = model.detect([right_img], verbose=0)[0]
        r_left = model.detect([left_img], verbose=0)[0]

        # For Right detection
        try:
            right_volume1 = r_right["masks"][:, :, 0].astype(bool)
            right_dice1 = calculate_dice(gt_right_mask[:, :, 0], right_volume1)
            try:
                right_volume2 = r_right["masks"][:, :, 1].astype(bool)
                right_dice2 = calculate_dice(gt_right_mask[:, :, 0], right_volume2)
                if right_dice1 >= right_dice2:
                    right_pred_mask = right_volume1
                    right_dice = right_dice1
                else:
                    right_pred_mask = right_volume2
                    right_dice = right_dice2
            except: 
                right_pred_mask = right_volume1
                right_dice = right_dice1
        except:
            print('No detection is made')
            right_pred_mask = np.zeros_like(image[:, :, 0])
            right_dice = 0

        # For Left detection
        try:
            left_volume1 = r_left["masks"][:, :, 0].astype(bool)
            left_dice1 = calculate_dice(gt_left_mask[:, :, 0], left_volume1)
            try:
                left_volume2 = r_left["masks"][:, :, 1].astype(bool)
                left_dice2 = calculate_dice(gt_left_mask[:, :, 0], left_volume2)
                if left_dice1 >= left_dice2:
                    left_pred_mask = left_volume1
                    left_dice = left_dice1
                else:
                    left_pred_mask = left_volume2
                    left_dice = left_dice2
            except: 
                left_pred_mask = left_volume1
                left_dice = left_dice1
        except:
            print('No detection is made')
            left_pred_mask = np.zeros_like(image[:, :, 0])
            left_dice = 0

        # Store the results in the DataFrame
        print('Right Dice: ', right_dice)
        print('Left Dice: ', left_dice)

        total_filename = filename + f'nIm_1'
        results_df = results_df.append({'Filename': total_filename, 
                                        'RightDiceScore': right_dice, 
                                        'LeftDiceScore': left_dice}, ignore_index=True)
        
        total_mask = right_pred_mask.astype('uint8') + left_pred_mask.astype('uint8')
        # total_mask_rgb = skimage.color.gray2rgb(total_mask)
        total_mask_resized = resize_image(sitk.GetImageFromArray(total_mask), (96,96), is_mask=True)
        mask_filepath = os.path.join(predictions_path + 'seg/', total_filename + '.nii')
        os.makedirs(predictions_path + 'seg/', exist_ok=True)
        sitk.WriteImage(total_mask_resized, mask_filepath)


    return results_df

def resize_image(image, new_size, is_mask=False):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [(original_spacing[i] * original_size[i] / new_size[i]) for i in range(len(new_size))]
    
    # Choose the interpolation method based on whether the image is a mask
    if is_mask:
        interpolation_method = sitk.sitkNearestNeighbor
    else:
        interpolation_method = sitk.sitkLinear
    
    resampled_image = sitk.Resample(image, new_size, sitk.Transform(), interpolation_method,
                                    image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                                    image.GetPixelID())
    return resampled_image

def preprocess_img(image, rescale = None, int_window = None): 
    image_resized = resize_image(image, (256, 256), is_mask=False)
    image_array = sitk.GetArrayFromImage(image_resized)
    # image_array.astype('float32')
    image_rgb = skimage.color.gray2rgb(image_array)
    if rescale:
        image_rescaled1 = sitk.RescaleIntensity(sitk.GetImageFromArray(image_rgb), 0, 255)
        if int_window:
            image_rescaled2 = preprocessing.specific_intensity_window_1(sitk.GetArrayFromImage(image_rescaled1), window_percent=0.15)
            return image_rescaled2
        else:
            return sitk.GetArrayFromImage(image_rescaled1)
    if int_window and not rescale:
        image_rescaled = preprocessing.specific_intensity_window_1((image_rgb), window_percent=0.15)
        return image_rescaled

    else:
        return image_rgb

def evaluate_GVox_tk(model, img_path, studies, predictions_path):

    results_df = pd.DataFrame(columns=['Filename', 'DiceScore'])
    dice_accumulated_df = pd.DataFrame(columns=['Study', 'DiceScoreAccumulated'])

    for i, nstudies in enumerate(studies):
        filename = nstudies.replace('/', '_')
        print('Evaluating study: ', filename)

        # Load image paths
        images_path = img_path + nstudies + 'with_M0/images/result.nii'
        masks_path = img_path + nstudies + 'with_M0/masks/Kidney/result.nii'

        images = sitk.ReadImage(images_path)
        gt_masks = sitk.ReadImage(masks_path) # toda la serie

        image = images[:, :, 0]
        gt_mask = gt_masks[:,:,0]
        image = preprocess_img(image, rescale=True, int_window=False)
        gt_mask = resize_image(gt_mask, (256, 256), is_mask=True)
        gt_mask = sitk.GetArrayFromImage(gt_mask)
        gt_mask = skimage.color.gray2rgb(gt_mask)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]

        # For Right detection
        try:
            volume1 = r["masks"][:, :, 0].astype(bool)
            dice1 = calculate_dice(gt_mask[:, :, 0], volume1)
            try:
                volume2 = r["masks"][:, :, 1].astype(bool)
                dice2 = calculate_dice(gt_mask[:, :, 0], volume2)
                if dice1 >= dice2:
                    pred_mask = volume1
                    dice = dice1
                else:
                    pred_mask = volume2
                    dice = dice2
            except: 
                pred_mask = volume1
                dice = dice1
        except:
            print('No detection is made')
            pred_mask = np.zeros_like(image[:, :, 0])
            dice = 0

        total_filename = filename + f'nIm_1'
        results_df = results_df.append({'Filename': total_filename, 
                                        'DiceScore': dice}, ignore_index=True)
        
        total_mask_resized = resize_image(sitk.GetImageFromArray(pred_mask.astype('uint8')), (96,96), is_mask=True)
        mask_filepath = os.path.join(predictions_path + 'seg/', total_filename + '.nii')
        os.makedirs(predictions_path + 'seg/', exist_ok=True)
        sitk.WriteImage(total_mask_resized, mask_filepath)

    return results_df

def evaluate_GVox_hk(model, img_path, studies, predictions_path):

    results_df = pd.DataFrame(columns=['Filename', 'RightDiceScore', 'LeftDiceScore'])
    dice_accumulated_df = pd.DataFrame(columns=['Study', 'DiceScoreRightAccumulated', 'DiceScoreLeftAccumulated'])

    for i, nstudies in enumerate(studies):
        filename = nstudies.replace('/', '_')
        print('Evaluating study: ', filename)

        # Load image paths
        images_path = img_path + nstudies + 'with_M0/images/result.nii'
        masks_path = img_path + nstudies + 'with_M0/masks/Kidney/result.nii'

        images = sitk.ReadImage(images_path)
        gt_masks = sitk.ReadImage(masks_path) # toda la serie

        image = images[:, :, 0]
        image = preprocess_img(image, rescale=True, int_window=False)
        image_resized = resize_image(sitk.GetImageFromArray(image), (3, 96, 96), is_mask=False)
        
        image_blurred = preprocessing.blur_half_img(sitk.GetArrayFromImage(image_resized))
        right_img = image_blurred[0, :, :, 0]
        right_img = preprocess_img(sitk.GetImageFromArray(right_img), rescale=True, int_window=False)
        left_img = image_blurred[0, :, :, 1]
        left_img = preprocess_img(sitk.GetImageFromArray(left_img), rescale=True, int_window=False)

        gt_mask = gt_masks[:, :, 0]
        gt_right_mask, gt_left_mask = preprocessing.label_right_left(sitk.GetArrayFromImage(gt_mask))
        gt_right_mask = preprocess_img(sitk.GetImageFromArray(gt_right_mask))
        gt_left_mask = preprocess_img(sitk.GetImageFromArray(gt_left_mask))

        # Run detection
        t = time.time()
        r_right = model.detect([right_img], verbose=0)[0]
        r_left = model.detect([left_img], verbose=0)[0]

        # For Right detection
        try:
            right_volume1 = r_right["masks"][:, :, 0].astype(bool)
            right_dice1 = calculate_dice(gt_right_mask[:, :, 0], right_volume1)
            try:
                right_volume2 = r_right["masks"][:, :, 1].astype(bool)
                right_dice2 = calculate_dice(gt_right_mask[:, :, 0], right_volume2)
                if right_dice1 >= right_dice2:
                    right_pred_mask = right_volume1
                    right_dice = right_dice1
                else:
                    right_pred_mask = right_volume2
                    right_dice = right_dice2
            except: 
                right_pred_mask = right_volume1
                right_dice = right_dice1
        except:
            print('No detection is made')
            right_pred_mask = np.zeros_like(image[:, :, 0])
            right_dice = 0

        # For Left detection
        try:
            left_volume1 = r_left["masks"][:, :, 0].astype(bool)
            left_dice1 = calculate_dice(gt_left_mask[:, :, 0], left_volume1)
            try:
                left_volume2 = r_left["masks"][:, :, 1].astype(bool)
                left_dice2 = calculate_dice(gt_left_mask[:, :, 0], left_volume2)
                if left_dice1 >= left_dice2:
                    left_pred_mask = left_volume1
                    left_dice = left_dice1
                else:
                    left_pred_mask = left_volume2
                    left_dice = left_dice2
            except: 
                left_pred_mask = left_volume1
                left_dice = left_dice1
        except:
            print('No detection is made')
            left_pred_mask = np.zeros_like(image[:, :, 0])
            left_dice = 0

        # Store the results in the DataFrame
        print('Right Dice: ', right_dice)
        print('Left Dice: ', left_dice)

        total_filename = filename + f'nIm_1'
        results_df = results_df.append({'Filename': total_filename, 
                                        'RightDiceScore': right_dice, 
                                        'LeftDiceScore': left_dice}, ignore_index=True)
        
        total_mask = right_pred_mask.astype('uint8') + left_pred_mask.astype('uint8')
        # total_mask_rgb = skimage.color.gray2rgb(total_mask)
        total_mask_resized = resize_image(sitk.GetImageFromArray(total_mask), (96,96), is_mask=True)
        mask_filepath = os.path.join(predictions_path + 'seg/', total_filename + '.nii')
        os.makedirs(predictions_path + 'seg/', exist_ok=True)
        sitk.WriteImage(total_mask_resized, mask_filepath)

    return results_df


if __name__ == '__main__': 

    # --------------------------------------------------------------------------------------------------------------------------
    # GENERAL SETTINGS ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # models_tk = ['TRACTRL02', 'TRACTRL03', 'TRACTRL05', 'TRACTRL06', 'TRACTRL08', 'TRACTRL09', 'TRACTRL10', 'TRACTRL11', 
    #               'TRACTRL13', 'TRACTRL14', 'TRACTRL15', 'TRACTRL17', 'TRACTRL18', 'TRACTRL19', 'TRACTRL20', 'TRACTRL21']
    models_tk = ['TRACTRL02', 'TRACTRL05', 'TRACTRL09', 'TRACTRL10', 'TRACTRL15', 
                 'TRACTRL18', 'TRACTRL19', 'TRACTRL20']
    weights_tk = ['0060', '0026', '0025', '0139', '0075', '0032', '0048', '0082']
    # weights_tk = ['0150', '0123', '0130', '0147', '0087', '0107', '0113', '0119']
    gvox_path = 'Z:/RM_RENAL/Registration/Voxelmorph/Results/NCC-DICE/w_0.9-0.1/U-l2/'

    
    for i, model_to_test in enumerate(models_tk):
    
        logs_dir = 'Z:/RM_RENAL/Data_Segmentation/Models/Baseline_e150_LayersAll/'
        test_tk = model_to_test
        # model_dir = logs_dir
        model_dir  = logs_dir + 'Test_' + test_tk + '/'
        weights_path = os.path.join(model_dir, 'mask_rcnn_kidney_' + weights_tk[i] + '.h5')

        mode = 'inference'
        config = InferenceConfig()
        model = modellib.MaskRCNN(
            mode=mode,
            config=config,
            model_dir=logs_dir, 
            )

        print('Loading weights...')
        tf.keras.Model.load_weights(model.keras_model, weights_path, by_name = True)

        # --------------------------------------------------------------------------------------------------------------------------
        # Evaluation on real transplanted kidney data (TK) ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        data_path_tk = 'Z:/RM_RENAL/DATA/Allograft/'
        images_tk = data_path_tk + 'images/'
        masks_tk = data_path_tk + 'masks/'

        with open('data_Allograft_GVox.txt') as f:   
            studies_GVox = f.read().splitlines() 
        
        with open('data_Allograft.txt') as f:   
            studies = f.read().splitlines() 

        predicted_path_tk = model_dir + 'Predictions_TK/'
        os.makedirs(predicted_path_tk, exist_ok=True)

        studies_filtered = [path for path in studies if test_tk in path]
        studies_filtered_gvox = [path for path in studies_GVox if test_tk in path]

        # results_tk = evaluate_tk(model, data_path_tk, studies_filtered, predicted_path_tk)
        # results_tk.to_excel(predicted_path_tk + 'dice_scores_rescaled.xlsx', index=False)

        results_GVox_tk, results_oneslice_GVox_tk = evaluate_GVox_tk(model, gvox_path + 'Allograft/', studies_filtered_gvox, predicted_path_tk)
        results_GVox_tk.to_excel(predicted_path_tk + 'dice_scores_GVox_M0.xlsx', index=False)
        results_oneslice_GVox_tk.to_excel(predicted_path_tk + 'dice_scores_GVox_OneSlice_M0.xlsx', index=False)

        # --------------------------------------------------------------------------------------------------------------------------
        # Evaluation on synthetic data (STK) ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        data_path_stk = 'Z:/RM_RENAL/CycleGAN/results/56modelXCAT/perceptual_loss/'
        exp = 'myExp_bs4_lr0.0002_l10_10_10_PL_0.5_BtoA_NoAug_unet256_resized256'
        images_stk = data_path_stk + exp + '/'
        masks_stk = 'Z:/RM_RENAL/CycleGAN/masks/'
            
        studies = ['93', '178']

        predicted_path_stk = model_dir + 'Predictions_STK/'
        os.makedirs(predicted_path_stk, exist_ok=True)

        results_stk = evaluate_stk(model, images_stk, masks_stk, studies, predicted_path_stk)
        results_stk.to_excel(os.path.join(predicted_path_stk, 'dice_results_stk_M0.xlsx'), index=False)


        # --------------------------------------------------------------------------------------------------------------------------
        # Evaluation on real healthy kidney (HK) data ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        img_path = 'Z:/RM_RENAL/Registration/Unregistered_Data/Native/'
        mask_path = 'Z:/RM_RENAL/DATA/Native/masks/'

        with open('data_Native_Kidney.txt') as f:   
            studies = f.read().splitlines() 
        
        predicted_path_hk = model_dir + 'Predictions_HK/'
        os.makedirs(predicted_path_hk, exist_ok=True)

        # results_hk = evaluate_hk(model, img_path, mask_path, studies, predicted_path_hk)
        # results_hk.to_excel(predicted_path_hk + 'dice_scores_M0.xlsx', index=False)

        results_GVox_hk, results_oneslice_GVox_hk = evaluate_GVox_hk(model, gvox_path + 'Native/', studies, predicted_path_tk)
        results_GVox_hk.to_excel(predicted_path_tk + 'dice_scores_GVox_M0.xlsx', index=False)
        results_oneslice_GVox_hk.to_excel(predicted_path_tk + 'dice_scores_GVox_OneSlice_M0.xlsx', index=False)
