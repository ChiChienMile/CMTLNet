# ANTsPy (https://github.com/ANTsX/ANTsPy)
import ants

def registration_function(fixed_image_path, moving_image_path, output_image_path):
    """
    Registers a moving image to a fixed image using affine transformation and saves the registered image.

    Parameters:
        fixed_image_path (str): Path to the fixed image (reference image).
        moving_image_path (str): Path to the moving image (image to be registered).
        output_image_path (str): Path where the registered image will be saved.
    """
    # Step 1: Load images using ANTs
    fixed_image = ants.image_read(fixed_image_path)  # Read the fixed (reference) image
    moving_image = ants.image_read(moving_image_path)  # Read the moving image

    '''
    ANTs registration function returns a dictionary containing:
        - warpedmovout: The moving image registered to the fixed image
        - warpedfixout: The fixed image registered to the moving image
        - fwdtransforms: Transformation field from moving to fixed image
        - invtransforms: Transformation field from fixed to moving image

    Available transform types:
        - Rigid: Rigid transformation (translation + rotation)
        - Affine: Affine transformation (rigid + scaling)
        - ElasticSyN: Elastic SyN transformation (affine + nonlinear with MI as the criterion)
        - SyN: SyN transformation (affine + nonlinear with MI as the criterion)
        - SyNCC: SyN transformation (affine + nonlinear with CC as the criterion)
    '''

    # Step 2: Perform image registration (using affine transformation)
    registration_result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Affine')

    # Step 3: Apply the forward transformation to the moving image to get the registered image
    registered_image = ants.apply_transforms(
        fixed=fixed_image,  # Fixed image (reference)
        moving=moving_image,  # Moving image (to be registered)
        transformlist=registration_result['fwdtransforms'],  # Transformation field
        interpolator="linear"  # Interpolation method for resampling
    )

    # Step 4: Ensure the registered image matches the fixed image's properties (direction, origin, spacing)
    registered_image.set_direction(fixed_image.direction)
    registered_image.set_origin(fixed_image.origin)
    registered_image.set_spacing(fixed_image.spacing)

    # Step 5: Save the registered image
    ants.image_write(registered_image, output_image_path)

    print("Registration complete. Registered image saved to:", output_image_path)


if __name__ == '__main__':
    # Define the paths for fixed and moving images, and the output paths for the registered images
    fixed_image_path = "./regdata/spgr_unstrip_lps_b.nii.gz"

    moving_image_path1 = "./regdata/LGG-637/T1C.nii.gz"
    output_image_path1 = "./regdata/LGG-637/registration_T1C.nii.gz"

    moving_image_path2 = "./regdata/LGG-637/T2W.nii.gz"
    output_image_path2 = "./regdata/LGG-637/registration_T2W.nii.gz"

    # Perform registration for the first moving image (T1C)
    registration_function(fixed_image_path, moving_image_path1, output_image_path1)

    # Perform registration for the second moving image (T2w) using the registered T1C image as the fixed image
    registration_function(output_image_path1, moving_image_path2, output_image_path2)