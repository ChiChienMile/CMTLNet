import os
import pickle
import nibabel as nib
import numpy as np


def load_pickle(file: str, mode: str = 'rb'):
    """
    Load and return a Python object from a pickle file.

    Parameters:
    - file (str): The path to the pickle file.
    - mode (str): The mode in which to open the file. Defaults to 'rb' (read binary).

    Returns:
    - The object loaded from the pickle file.
    """
    with open(file, mode) as f:
        a = pickle.load(f)  # Load the object from the pickle file
    return a  # Return the loaded object

def readDir(path, num):
    """
    Read a specific subdirectory from the given path.

    Parameters:
    - path (str): The directory path to read.
    - num (int): The index of the subdirectory to retrieve.

    Returns:
    - im (list): A list containing the name of the specified subdirectory.
    """
    im = []
    k = 0  # Initialize a counter
    for root in os.walk(path):  # Traverse the directory tree
        if k == 0:
            im.append(root[num])  # Append the nth subdirectory
            k = k + 1  # Increment the counter to avoid further processing
        if k == 1:
            continue  # Skip processing other directories
    return im  # Return the list containing the desired subdirectory


def load_nii_affine(filename):
    """
    Load a NIfTI file and return its data and affine matrix.

    Parameters:
    - filename (str): The path to the NIfTI file.

    Returns:
    - data (numpy.ndarray): The image data from the NIfTI file.
    - affine (numpy.ndarray): The affine transformation matrix of the image.

    If the file does not exist, returns a numpy array containing [1].
    """
    if not os.path.exists(filename):
        return np.array([1])  # Return a default array if file does not exist
    nii = nib.load(filename)  # Load the NIfTI file
    data = nii.get_fdata()  # Retrieve the image data as a floating-point array
    affine = nii.affine  # Retrieve the affine transformation matrix
    nii.uncache()  # Uncache the NIfTI object to free memory
    return data, affine  # Return the data and affine matrix

def get_rectangle_3d(mask):
    """
    Determine the bounding box coordinates of the non-zero region in a 3D mask.

    Parameters:
    - mask (numpy.ndarray): A 3D mask array where non-zero values indicate the region of interest.

    Returns:
    - min_x, max_x, min_y, max_y, min_z, max_z (int): The minimum and maximum indices along each axis that contain non-zero values.
    """
    mask_size = mask.shape  # Get the shape of the mask
    min_x, min_y, min_z = mask_size[0], mask_size[1], mask_size[2]  # Initialize minimum indices
    max_x, max_y, max_z = 0, 0, 0  # Initialize maximum indices

    # Iterate through each voxel in the mask to find the bounding box
    for index_i in range(mask_size[0]):
        for index_j in range(mask_size[1]):
            for index_k in range(mask_size[2]):
                if mask[index_i][index_j][index_k] > 0:  # Check if the voxel is part of the region of interest
                    # Update minimum indices
                    if min_x > index_i:
                        min_x = index_i
                    if min_y > index_j:
                        min_y = index_j
                    if min_z > index_k:
                        min_z = index_k

                    # Update maximum indices
                    if max_x < index_i:
                        max_x = index_i
                    if max_y < index_j:
                        max_y = index_j
                    if max_z < index_k:
                        max_z = index_k

    return min_x, max_x, min_y, max_y, min_z, max_z  # Return the bounding box coordinates


def Padding3d(data, padding_shape=(160, 192, 160)):
    """
    Pad a 3D array to the specified shape with zeros, centering the original data.

    Parameters:
    - data (numpy.ndarray): The 3D array to pad.
    - padding_shape (tuple): The desired output shape after padding. Defaults to (160, 192, 160).

    Returns:
    - data (numpy.ndarray): The padded 3D array.
    """
    # Calculate padding sizes for each dimension
    padding_h, padding_w, padding_d = (np.array(padding_shape) - np.array(data.shape)) // 2

    # Create padding arrays for the top and bottom along the first axis (height)
    padding_T = np.zeros([padding_h, data.shape[1], data.shape[2]], dtype=np.float32)  # Top padding
    padding_B = np.zeros([padding_shape[0] - padding_h - data.shape[0],
                          data.shape[1], data.shape[2]], dtype=np.float32)  # Bottom padding
    H_data = np.concatenate([padding_T, data, padding_B], axis=0)  # Concatenate along the height axis

    # Create padding arrays for the left and right along the second axis (width)
    padding_L = np.zeros([H_data.shape[0], padding_w, H_data.shape[2]], dtype=np.float32)  # Left padding
    padding_R = np.zeros([H_data.shape[0],
                          padding_shape[1] - padding_w - H_data.shape[1], H_data.shape[2]], dtype=np.float32)  # Right padding
    W_data = np.concatenate([padding_L, H_data, padding_R], axis=1)  # Concatenate along the width axis

    # Create padding arrays for the front and back along the third axis (depth)
    padding_F = np.zeros([W_data.shape[0], W_data.shape[1], padding_d], dtype=np.float32)  # Front padding
    padding_Ba = np.zeros([W_data.shape[0], W_data.shape[1],
                           padding_shape[2] - padding_d - W_data.shape[2]], dtype=np.float32)  # Back padding
    data_padded = np.concatenate([padding_F, W_data, padding_Ba], axis=2)  # Concatenate along the depth axis

    return data_padded  # Return the padded data


def random_crop(img_T1C, img_T2W, mask_input_img_T1C, mask_input_img_T2W, mask, force_fg=True):
    """
    Perform a random crop on the provided 3D images and masks. With a probability of 0.8,
    the function crops the images and masks. If force_fg is True, the crop will include
    the entire region of interest defined by the mask.

    Parameters:
    - img_T1C (numpy.ndarray): The T1-weighted contrast-enhanced image.
    - img_T2W (numpy.ndarray): The T2-weighted image.
    - mask_input_img_T1C (numpy.ndarray): The mask corresponding to img_T1C.
    - mask_input_img_T2W (numpy.ndarray): The mask corresponding to img_T2W.
    - mask (numpy.ndarray): The primary mask indicating the region of interest (e.g., tumor).
    - force_fg (bool): Whether to force the crop to include the foreground (region of interest). Defaults to True.

    Returns:
    - tuple: A tuple containing the cropped (and possibly padded) images and masks:
        - crop_img_T1C (numpy.ndarray)
        - crop_img_T2W (numpy.ndarray)
        - mask_input_img_T1C (numpy.ndarray)
        - mask_input_img_T2W (numpy.ndarray)
        - crop_mask (numpy.ndarray)

    If the random probability check fails (20% chance), the original images and masks are returned.
    """
    P = np.random.random()  # Generate a random float between 0 and 1
    if P < 0.8:  # 80% chance to perform cropping
        shape_im = img_T1C.shape  # Get the shape of the input images
        crop_size = shape_im  # Set the crop size to the entire image size

        # Calculate lower and upper bounds for cropping along each axis
        lb_x = 0
        ub_x = shape_im[0] - crop_size[0] // 2
        lb_y = 0
        ub_y = shape_im[1] - crop_size[1] // 2
        lb_z = 0
        ub_z = shape_im[2] - crop_size[2] // 2

        if force_fg:
            # Get the bounding box coordinates of the region of interest from the mask
            P_minx, P_max, P_miny, P_maxy, P_minz, P_maxz = get_rectangle_3d(mask)
            # Randomly choose the lower bound of the crop to ensure it includes the entire tumor region
            bbox_x_lb = np.random.randint(0, P_minx + 1)
            bbox_y_lb = np.random.randint(0, P_miny + 1)
            bbox_z_lb = np.random.randint(0, P_minz + 1)
        else:
            # Otherwise, randomly choose the lower bound within the specified range
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

        # Define the upper bounds of the crop based on the lower bounds and crop size
        bbox_x_ub = bbox_x_lb + crop_size[0]
        bbox_y_ub = bbox_y_lb + crop_size[1]
        bbox_z_ub = bbox_z_lb + crop_size[2]

        # Adjust the crop boundaries to ensure they are within the image dimensions
        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape_im[0], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape_im[1], bbox_y_ub)
        valid_bbox_z_lb = max(0, bbox_z_lb)
        valid_bbox_z_ub = min(shape_im[2], bbox_z_ub)

        # Crop the T1C image
        crop_img_T1C = np.copy(img_T1C[valid_bbox_x_lb:valid_bbox_x_ub,
                                       valid_bbox_y_lb:valid_bbox_y_ub,
                                       valid_bbox_z_lb:valid_bbox_z_ub])

        # Crop the T2W image
        crop_img_T2W = np.copy(img_T2W[valid_bbox_x_lb:valid_bbox_x_ub,
                                       valid_bbox_y_lb:valid_bbox_y_ub,
                                       valid_bbox_z_lb:valid_bbox_z_ub])

        # Crop the mask corresponding to img_T1C
        mask_input_img_T1C = np.copy(mask_input_img_T1C[valid_bbox_x_lb:valid_bbox_x_ub,
                                                       valid_bbox_y_lb:valid_bbox_y_ub,
                                                       valid_bbox_z_lb:valid_bbox_z_ub])

        # Crop the mask corresponding to img_T2W
        mask_input_img_T2W = np.copy(mask_input_img_T2W[valid_bbox_x_lb:valid_bbox_x_ub,
                                                       valid_bbox_y_lb:valid_bbox_y_ub,
                                                       valid_bbox_z_lb:valid_bbox_z_ub])

        # Crop the primary mask
        crop_mask = np.copy(mask[valid_bbox_x_lb:valid_bbox_x_ub,
                                  valid_bbox_y_lb:valid_bbox_y_ub,
                                  valid_bbox_z_lb:valid_bbox_z_ub])

        # Pad the cropped images and masks to restore them to the original shape
        crop_img_T1C = Padding3d(crop_img_T1C, shape_im)
        crop_img_T2W = Padding3d(crop_img_T2W, shape_im)
        mask_input_img_T1C = Padding3d(mask_input_img_T1C, shape_im)
        mask_input_img_T2W = Padding3d(mask_input_img_T2W, shape_im)
        crop_mask = Padding3d(crop_mask, shape_im)

        # Return the cropped (and padded) images and masks
        return crop_img_T1C, crop_img_T2W, mask_input_img_T1C, mask_input_img_T2W, crop_mask
    else:
        # With 20% probability, return the original images and masks without cropping
        return img_T1C, img_T2W, mask_input_img_T1C, mask_input_img_T2W, mask


def load_mask(path):
    """
    Load the mask image from the specified path.

    Parameters:
    - path (str): The directory path where 'mask.nii' is located.

    Returns:
    - mask (numpy.ndarray): The mask data as a floating-point array.
    """
    mask_img_path = os.path.join(path, 'mask.nii')  # Construct the mask file path
    mask, _ = load_nii_affine(mask_img_path)  # Load the mask data and affine matrix
    mask = (mask > 0)
    mask = np.array(mask, dtype=float)  # Convert the mask to a float array
    return mask  # Return the mask


def load_img(path, type):
    """
    Load and preprocess an image from a NIfTI file.

    Parameters:
    - path (str): The directory path where the NIfTI file is located.
    - type (str): The base name of the NIfTI file (without the '.nii' extension).

    Returns:
    - img (numpy.ndarray): The normalized image data.
    - img_mask (numpy.ndarray): A binary mask indicating non-zero regions in the image.
    - affine (numpy.ndarray): The affine transformation matrix of the image.
    """
    # Construct the full file path by combining the directory path and file type
    full_path = os.path.join(path, f'{type}.nii')
    img, affine = load_nii_affine(full_path)  # Load the image data and affine matrix
    img = np.array(img, dtype=float)  # Ensure the image data is in floating-point format

    # Create a binary mask where pixels with values greater than 0 are marked as True
    img_mask = (img > 0)

    # Define a small constant to prevent division by zero during normalization
    smooth = 1e-8

    # Normalize the image data: subtract the mean and divide by the standard deviation
    img[img_mask] = (img[img_mask] - img[img_mask].mean()) * 1.0 / (img[img_mask].std() + smooth)

    # Convert the binary mask to a floating-point array
    img_mask = np.array(img_mask, dtype=float)

    return img, img_mask, affine  # Return the normalized image, mask, and affine matrix


def get_transimg(img_T1C_in, img_T2W_in, mask_input_T1C_in, mask_input_T2W_in, mask_seg_in):
    """
    Perform random cropping and prepare image and mask tensors for training.

    Args:
        img_T1C_in (numpy.ndarray): T1C modality image.
        img_T2W_in (numpy.ndarray): T2W modality image.
        mask_input_T1C_in (numpy.ndarray): Mask for T1C image.
        mask_input_T2W_in (numpy.ndarray): Mask for T2W image.
        mask_seg_in (numpy.ndarray): Segmentation mask.

    Returns:
        tuple: Processed images and masks.
    """
    # Apply random cropping to all inputs
    img_T1C, img_T2W, mask_input_T1C, mask_input_T2W, mask_seg = random_crop(
        img_T1C_in, img_T2W_in,
        mask_input_T1C_in, mask_input_T2W_in,
        mask_seg_in
    )
    # Combine input masks to create a unified mask
    mask_input_combined = (mask_input_T1C + mask_input_T2W) > 0
    mask_input_combined = mask_input_combined.astype(float)

    # Create original, positive, and negative masks based on segmentation
    mask_original = np.ones_like(mask_input_combined) * mask_input_combined
    mask_positive = mask_seg * mask_input_combined
    mask_negative = (1 - mask_seg) * mask_input_combined

    # Prepare image data as a list of modalities
    images = [img_T1C, img_T2W]
    # Prepare segmentation masks
    masks = [mask_original, mask_positive, mask_negative]
    return images, masks


def default_loader(path):
    """
    Load and process training images and masks from a given path.

    Args:
        path (str): Path to the image data.

    Returns:
        tuple: Processed images and masks for training.
    """
    # Load T1C and T2W images with their masks and affine transformations
    img_T1C, mask_input_T1C, affine = load_img(path, 'T1C')
    img_T2W, mask_input_T2W, _ = load_img(path, 'T2')

    # Load segmentation mask and convert to binary format
    mask_seg = load_mask(path)
    mask_seg = (mask_seg > 0).astype(float)

    # Transform images and masks using the get_transimg function
    imgs_k_img, imgs_k_mask = get_transimg(
        img_T1C, img_T2W,
        mask_input_T1C, mask_input_T2W,
        mask_seg
    )
    return imgs_k_img, imgs_k_mask


def default_loader_test(path):
    """
    Load and process test images and masks from a given path.

    Args:
        path (str): Path to the image data.

    Returns:
        tuple: Processed images and masks for testing.
    """
    # Load T1C and T2W images with their masks and affine transformations
    img_T1C, mask_input_T1C, affine = load_img(path, 'T1C')
    img_T2W, mask_input_T2W, _ = load_img(path, 'T2')

    # Combine input masks to create a unified mask
    mask_input_combined = (mask_input_T1C + mask_input_T2W) > 0
    mask_input_combined = mask_input_combined.astype(float)

    # Create original mask based on the combined input mask
    mask_original = np.ones_like(mask_input_combined) * mask_input_combined

    # Prepare image data as a list of modalities
    images = [img_T1C, img_T2W]
    # Prepare segmentation masks
    masks = [mask_original]
    return images, masks