import os
import numpy as np
import nibabel as nib
from os import path

# Read files from the specified directory at a given level (num).
def readDir(path, num):
    """
    Reads the directory and returns the filenames from the specified level (num).
    """
    im = []
    k = 0
    for root in os.walk(path):
        if k == 0:
            im.append(root[num])
            k = k+1
        if k == 1:
            continue
    return im

# Recursively get all filenames in a directory and its subdirectories.
def get_filenames(dirpath, filenames):
    """
    Recursively get all filenames in a directory and its subdirectories.
    """
    if not path.isabs(dirpath):
        dirpath = path.abspath(dirpath)
    for pathname, dirs, files in os.walk(dirpath):
        if files:
            for f in files:
                filenames.append(path.join(pathname, f))
        if dirs:
            for dir_ in dirs:
                get_filenames(path.join(pathname, dir_), filenames)

# Save a numpy array as a NIfTI image
def save_nii(arr, file_path, affine):
    """
    Save a numpy array as a NIfTI image.
    """
    nib.Nifti1Image(arr, affine).to_filename(file_path)

# Load a NIfTI file and return its data and affine transformation matrix
def load_nii_affine(filename):
    """
    Loads a NIfTI file and returns its data and affine transformation matrix.
    """
    if not os.path.exists(filename):
        return np.array([1])
    nii = nib.load(filename)
    data = nii.get_fdata()  # get_fdata() is preferred over get_data() in newer nibabel versions
    affine = nii.affine
    nii.uncache()
    return data, affine

# Calculate the bounding box of a mask (region of interest)
def get_bbox_from_mask(mask, outside_value=0):
    """
    Calculate the bounding box of a mask (region of interest).
    """
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]

# Crop the image to the given bounding box
def crop_to_bbox(image, bbox):
    """
    Crop the image to the given bounding box.
    """
    assert len(image.shape) == 3, "Only supports 3D images."
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

# Extract and crop the data from images
def Crop_data():
    """
    Crop the images in the specified directory and save the cropped images to the target path.
    """
    import warnings
    warnings.filterwarnings("ignore")
    all_file_names = readDir(original_img_path, 1)[0]

    img_size_list = []
    for img_index in range(len(all_file_names)):
        print(f'== Processing {img_index + 1}/{len(all_file_names)}: {all_file_names[img_index]} ==')

        # File paths
        img_path_T1 = original_img_path + all_file_names[img_index] + '/' + all_file_names[img_index] + '_t1.nii.gz'
        img_path_T1C = original_img_path + all_file_names[img_index] + '/' + all_file_names[img_index] + '_t1ce.nii.gz'
        img_path_T2 = original_img_path + all_file_names[img_index] + '/' + all_file_names[img_index] + '_t2.nii.gz'
        img_path_FLAIR = original_img_path + all_file_names[img_index] + '/' + all_file_names[img_index] + '_flair.nii.gz'
        img_path_seg = original_img_path + all_file_names[img_index] + '/' + all_file_names[img_index] + '_seg.nii.gz'

        # Load data
        nii_data_1, affine = load_nii_affine(img_path_T1)
        nii_data_2, affine_2 = load_nii_affine(img_path_T1C)
        nii_data_3, affine_3 = load_nii_affine(img_path_T2)
        nii_data_4, affine_4 = load_nii_affine(img_path_FLAIR)
        nii_seg, affine_seg = load_nii_affine(img_path_seg)

        # Calculate bounding box
        combined_image = nii_data_1 + nii_data_2 + nii_data_3 + nii_data_4
        bbox = get_bbox_from_mask(combined_image)

        # Crop images
        cropped_1 = crop_to_bbox(nii_data_1, bbox)
        cropped_2 = crop_to_bbox(nii_data_2, bbox)
        cropped_3 = crop_to_bbox(nii_data_3, bbox)
        cropped_4 = crop_to_bbox(nii_data_4, bbox)
        cropped_seg = crop_to_bbox(nii_seg, bbox)
        cropped_seg = (cropped_seg > 0).astype(np.uint8)

        # Get cropped image size
        img_size = cropped_1.shape

        # Save paths
        save_path_T1 = save_img_path + all_file_names[img_index] + '/T1.nii'
        save_path_T1C = save_img_path + all_file_names[img_index] + '/T1C.nii'
        save_path_T2 = save_img_path + all_file_names[img_index] + '/T2.nii'
        save_path_FLAIR = save_img_path + all_file_names[img_index] + '/FLAIR.nii'
        save_path_seg = save_img_path + all_file_names[img_index] + '/mask.nii'

        if not os.path.exists(save_img_path + all_file_names[img_index] + '/'):
            os.makedirs(save_img_path + all_file_names[img_index] + '/')

        # Save cropped images
        save_nii(cropped_1.astype(np.uint16), save_path_T1, affine)
        save_nii(cropped_2.astype(np.uint16), save_path_T1C, affine)
        save_nii(cropped_3.astype(np.uint16), save_path_T2, affine)
        save_nii(cropped_4.astype(np.uint16), save_path_FLAIR, affine)
        save_nii(cropped_seg.astype(np.uint8), save_path_seg, affine_seg)

        img_size_list.append(img_size)

    # Save the crop sizes of all images
    np.save(save_crop_size_path + 'crop_size.npy', img_size_list)
    return all_file_names

def get_max_size():
    img_size_list = np.load(save_crop_size_path + 'crop_size.npy')
    size_1 = np.zeros((len(img_size_list)))
    size_2 = np.zeros((len(img_size_list)))
    size_3 = np.zeros((len(img_size_list)))
    for size_temp in range(len(img_size_list)):
        size_1[size_temp] = int(img_size_list[size_temp][0])
        size_2[size_temp] = int(img_size_list[size_temp][1])
        size_3[size_temp] = int(img_size_list[size_temp][2])
    print(np.max(size_1), np.max(size_2), np.max(size_3))

# 3D padding
def Padding3d(data, padding_shape):
    """
    Add padding to a 3D image.
    """
    padding_h, padding_w, padding_d = (np.array(padding_shape) - np.array(np.shape(data))) // 2

    padding_T = np.zeros([padding_h, np.shape(data)[1], np.shape(data)[2]], dtype=np.uint8)
    padding_B = np.zeros([padding_shape[0] - padding_h - np.shape(data)[0],
                          np.shape(data)[1], np.shape(data)[2]], dtype=np.uint8)
    H_data = np.concatenate([padding_T, data, padding_B], 0)

    padding_L = np.zeros([np.shape(H_data)[0], padding_w, np.shape(H_data)[2]], dtype=np.uint8)
    padding_R = np.zeros([np.shape(H_data)[0],
                          padding_shape[1] - padding_w - np.shape(H_data)[1], np.shape(H_data)[2]], dtype=np.uint8)
    W_data = np.concatenate([padding_L, H_data, padding_R], 1)

    padding_F = np.zeros([np.shape(W_data)[0], np.shape(W_data)[1], padding_d], dtype=np.uint8)
    padding_B = np.zeros([np.shape(W_data)[0], np.shape(W_data)[1], padding_shape[2] - padding_d - np.shape(W_data)[2]],
                         dtype=np.uint8)
    data = np.concatenate([padding_F, W_data, padding_B], 2)

    return data

# Pad the images with zeros and save them to the target directory.
def Padding_data(all_file_names, padding_shape=(160, 192, 160)):
    """
    Pad the images with zeros and save them to the target directory.
    """
    img_size_list = np.load(save_crop_size_path + 'crop_size.npy', allow_pickle=True)
    img_index = 0

    for img_size in img_size_list:
        print(f'== Padding {img_index + 1}/{len(img_size_list)} ==')

        # Get file paths
        img_path_T1 = save_img_path + all_file_names[img_index] + '/T1.nii'
        img_path_T1C = save_img_path + all_file_names[img_index] + '/T1C.nii'
        img_path_T2 = save_img_path + all_file_names[img_index] + '/T2.nii'
        img_path_FLAIR = save_img_path + all_file_names[img_index] + '/FLAIR.nii'
        img_path_seg = save_img_path + all_file_names[img_index] + '/mask.nii'

        # Load data
        nii_data_1, affine = load_nii_affine(img_path_T1)
        nii_data_2, affine_2 = load_nii_affine(img_path_T1C)
        nii_data_3, affine_3 = load_nii_affine(img_path_T2)
        nii_data_4, affine_4 = load_nii_affine(img_path_FLAIR)
        nii_seg, affine_seg = load_nii_affine(img_path_seg)

        # Perform padding
        padded_1 = Padding3d(nii_data_1, padding_shape)
        padded_2 = Padding3d(nii_data_2, padding_shape)
        padded_3 = Padding3d(nii_data_3, padding_shape)
        padded_4 = Padding3d(nii_data_4, padding_shape)
        padded_seg = Padding3d(nii_seg, padding_shape)

        if not os.path.exists(save_img_path_final + all_file_names[img_index] + '/'):
            os.makedirs(save_img_path_final + all_file_names[img_index] + '/')

        # Save padded images
        save_nii(padded_1.astype(np.uint16), save_img_path_final + all_file_names[img_index] + '/T1.nii', affine)
        save_nii(padded_2.astype(np.uint16), save_img_path_final + all_file_names[img_index] + '/T1C.nii', affine_2)
        save_nii(padded_3.astype(np.uint16), save_img_path_final + all_file_names[img_index] + '/T2.nii', affine_3)
        save_nii(padded_4.astype(np.uint16), save_img_path_final + all_file_names[img_index] + '/FLAIR.nii', affine_4)
        save_nii(padded_seg.astype(np.uint8), save_img_path_final + all_file_names[img_index] + '/mask.nii', affine_seg)

        img_index += 1

if __name__ == '__main__':
    # File path definitions
    original_img_path = './cropdata/beforedata/'  # Path to the original images
    save_img_path = './cropdata/aftercrop/'  # Path to save cropped images
    save_crop_size_path = './cropdata/'  # Path to save crop sizes
    save_img_path_final = './cropdata/afterpadding/'  # Path to save padded images

    # Step 1: Crop out the black borders from the data
    all_file_names = Crop_data()  # Calls the Crop_data function to crop images

    # Step 2: Verify if the maximum size in the data is smaller than (160, 192, 160)
    get_max_size()  # Calls the get_max_size function to check the maximum size in the data

    # Step 3: Add padding with zero values and zero slices to ensure each axis is a multiple of 3 or 2
    # The default padding size is (160, 192, 160)
    Padding_data(all_file_names)  # Calls the Padding_data function to add padding to the images

