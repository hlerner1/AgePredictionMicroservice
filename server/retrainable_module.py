from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2
import os
import numpy as np
from data_preprocessor import TransformDataset
from functools import reduce
from tensorflow.keras.utils import to_categorical
from rectified_adam import RAdam


def get_mapped_list(path):
    return [str(file) for r, _, f in os.walk(path) for file in f if any(map(str(file).lower().__contains__, ['.png', '.jpg', '.jpeg']))]  

def resize_image(img_path, output_shape):
    im = imread(img_path)
    resized = cv2.resize(im, (output_shape, output_shape), interpolation=cv2.INTER_AREA)
    return cv2.normalize(resized, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, 
                                 dtype = cv2.CV_32F).astype(np.uint8)

def resize_directory(base_path, file_lst, output_shape=160):
    for im_file in file_lst:
        rel_path = os.path.join(base_path, im_file)
        try:
            if imread(rel_path).shape == (output_shape, output_shape, 3):
                continue
            resized_im = resize_image(rel_path, output_shape)
            resized_im[resized_im<0] = 0
            plt.imsave(os.path.join(base_path, im_file), resized_im)
            if imread(os.path.join(base_path, im_file)).shape == (output_shape, output_shape, 4):
                resized_im = imread(rel_path)[:, :, :3]
                resized_im[resized_im<0] = 0
                plt.imsave(os.path.join(base_path, im_file), resized_im)
        except:
            os.remove(rel_path)
    return

def create_augmentation_set(original_path, transformations, severity):
    output_path = original_path + '_augmentation'
    file_lst = get_mapped_list(original_path)
    func = TransformDataset()
    has_severity = ['rotate_np', 'flip_rotate', 'perform_swirl_transformation', 'perform_random_affine_transform', 
                    'add_multiplicative_noise', 'add_shot_noise', 'add_gaussian_noise', 'add_impulse_noise', 
                    'add_glass_blur', 'add_gaussian_blur']
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for im_file in file_lst:
        im = imread(os.path.join(original_path, im_file))
        file_name, extension = im_file.split('.')
        
        for transF in transformations:
            if transF not in has_severity:
                try:
                    plt.imsave(os.path.join(output_path, ".".join(["_".join([file_name, transF]), extension])), 
                               func.return_function(transF, im))
                except:
                    print ("Failed to augment file = {}".format(file_name))
                    continue
            else:
                for severity_ in severity:
                    try:
                        plt.imsave(os.path.join(output_path, ".".join(["_".join([file_name, transF]), extension])), 
                           func.return_function(transF, im, severity_))
                    except:
                        print ("Failed to augment file = {}".format(file_name))
                        continue
        #print ("Successfully created Augmentation Set for {}".format(im_file))
    return output_path

def process_prepare_directories(dir_, transformations, label, severity, create_augmentation=True, 
                                data_map=[], paths_map=[], label_map=[]):
    dataset = get_mapped_list(dir_)
    resize_directory(dir_, dataset)
    
    data_map.append(dataset)
    paths_map.append(dir_)
    label_map.append(label)
    
    if create_augmentation:
        dir_augmented = create_augmentation_set(dir_, transformations, severity)
        dataset_augmented = get_mapped_list(dir_augmented)
        data_map.append(dataset_augmented)
        paths_map.append(dir_augmented)
        label_map.append(label)
        
    return data_map, paths_map, label_map

def create_dataset(data_map, path_map, label_map, output_shape):
    h, w, c = output_shape, output_shape, 3
    size_data = reduce(lambda x,y: x+y, map(lambda x: len(x), data_map))
    X = np.zeros((size_data, h, w, c), dtype=np.uint8)
    Y = np.zeros((size_data))
    x_ptr = 0
    for idx in range(len(data_map)):
        if label_map[idx]:
            Y[x_ptr:x_ptr+len(data_map[idx])] = np.ones(len(data_map[idx]))
        for file in data_map[idx]:
            im = imread(os.path.join(path_map[idx], file)).astype(np.uint8)
            if im.shape != (h, w, c):
                im = im[:, :, :c]
            X[x_ptr] = im
            x_ptr += 1
    return X, Y

def prepare_dataset_for_retraining (dir_coke, dir_not_coke, default_transformations_coke, 
                                    default_transformations_not_coke, output_shape, severity, create_augmentation=True):
    if len(dir_coke) == 0 and len(dir_not_coke) == 0:
        return None, None
    
    if len(default_transformations_coke) == 0:
        default_transformations_coke = ['flip_vertical_np', 'flip_horizontal_np', 'rotate_np', 'flip_rotate', 
        'perform_swirl_transformation', 'perform_random_affine_transform', 'add_multiplicative_noise', 'add_shot_noise', 
        'add_gaussian_noise', 'add_impulse_noise', 'add_glass_blur', 'add_gaussian_blur', 'random_image_eraser']
    if len(default_transformations_not_coke) == 0:
        default_transformations_not_coke = ['flip_vertical_np', 'flip_horizontal_np', 'rotate_np', 'flip_rotate', 
        'perform_swirl_transformation', 'perform_random_affine_transform', 'add_multiplicative_noise', 'add_shot_noise', 
        'add_gaussian_noise', 'add_impulse_noise', 'add_glass_blur', 'add_gaussian_blur', 'random_image_eraser']
        
    all_data, all_paths, is_label = process_prepare_directories(dir_=dir_coke, transformations=default_transformations_coke, 
        label=True, severity=severity, create_augmentation=create_augmentation, data_map=[], paths_map=[], label_map=[])
    all_data, all_paths, is_label = process_prepare_directories(dir_=dir_not_coke, 
        transformations=default_transformations_not_coke, label=False, severity=severity, create_augmentation=create_augmentation, 
        data_map=all_data, paths_map=all_paths, label_map=is_label)
    
    X, Y = create_dataset(all_data, all_paths, is_label, output_shape)
    return X, Y

def retrain_model(model, dir_coke, dir_not_coke, model_params, dir_val_coke, dir_val_not_coke, default_transformations_coke, 
          default_transformations_not_coke, augmentation_training, augmentation_validation, output_shape, severity):
    
    # Creating the training dataset
    X_train, y_train = prepare_dataset_for_retraining(dir_coke, dir_not_coke, default_transformations_coke, 
                                  default_transformations_not_coke, output_shape, severity, augmentation_training)
    y_train = to_categorical(y_train, num_classes=2)
    
    if len(dir_val_coke) != 0 or len(dir_val_not_coke) !=0: # Use only when either of the path is specified
        # Creating the validation dataset
        X_val, y_val = prepare_dataset_for_retraining(dir_val_coke, dir_val_not_coke,
                 default_transformations_coke, default_transformations_not_coke, output_shape, severity, augmentation_validation)
    
        y_val = to_categorical(y_val, num_classes=2)
    
    # Default Model parameters
    default_max_epochs, default_split_size, default_batch_size = 15, 6144, 3072
    default_model_save_path = 'sodanet/model'
    
    max_epochs = model_params.get('max_epochs') or default_max_epochs
    split_size = model_params.get('split_size') or default_split_size
    batch_size = model_params.get('batch_size') or default_batch_size
    model_path = model_params.get('external_model_path') or ''
    
    idxs = np.arange(X_train.shape[0])
    for chunk in np.array_split(idxs, int(np.ceil(X_train.shape[0] / split_size))):
        X_current, y_current = X_train[chunk], y_train[chunk]
        if len(dir_val_coke) != 0 or len(dir_val_not_coke) !=0:
            model.fit(X_current, y_current, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        else:
            model.fit(X_current, y_current, epochs=max_epochs, batch_size=batch_size)
    
        if len(model_path) > 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            model.save(os.path.join(model_path, 'AlexNet.hdf5'))
        else:
            model.save(os.path.join(default_model_save_path, 'AlexNet.hdf5'))
        
    return model