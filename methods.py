# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:01:59 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

Utility modules used in this projects 
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd


def calc_binObject_params(bin_img, maxObj_height, maxObj_width, maxObj2bg=True, plot=False):
    """
    Calculate parameters for binary objects in an image.

    Args:
        bin_img (ndarray): A binary image containing labeled objects.
        maxObj_height (int): The maximum height of objects to consider.
        maxObj_width (int): The maximum width of objects to consider.
        maxObj2bg (bool, optional): If True, objects that exceed the maximum dimensions will be set to background (default is True).
        plot (bool, optional): If True, a plot of the labeled objects will be displayed (default is False).

    Returns:
        list: A list of dictionaries, where each dictionary contains the parameters of a valid object.
        ndarray: The modified labeled image with objects removed if maxObj2bg is True.

    The function takes a binary image with labeled objects, where objects are represented by connected components.
    It calculates various parameters for each object, such as label, center coordinates, bounding box height and width,
    and the coordinates of the top-left and bottom-right corners of the bounding box. Objects that exceed the specified
    maximum height and width are optionally removed from the labeled image if maxObj2bg is set to True.

    If the plot argument is True, the function will also display a plot of the labeled objects in the image.

    Example usage:
    object_params, modified_img = calc_binObject_params(binary_image, 100, 100, maxObj2bg=True, plot=True)
    """
    labels, num_features = nd.label(bin_img)
    object_parameters = []
    
    if plot:
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(12,6))
        # Display the binary image
        ax.imshow(bin_img, cmap='gray')

    
    for label in range(0, num_features + 1):
        parameters = {}
        # Find coordinates of the labeled component
        coords = np.column_stack(np.where(labels == label))
        # Calculate the bounding box coordinates
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        
        # Calculate width and height
        width = x2 - x1 + 1 
        height = y2 - y1 + 1
        
        if width < maxObj_width and height < maxObj_height:
            # Calculate the center of the rectangle
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
    
            parameters['label'] = label
            parameters['center'] = (center_x, center_y)
            parameters['bboxHeight'] = height
            parameters['bboxWidth'] = width
            parameters['bboxTL'] = (x1, y1)
            parameters['bboxBR'] = (x2, y2)
    
            object_parameters.append(parameters)
            if plot:
                rect = plt.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        
        else:
            if maxObj2bg:
                labels[labels == label] = 0
    
    if plot:
        plt.show()
    
    return object_parameters, labels



def detect_patch_with_object(imarray, labels, patch_size=256, step_size=256, plot=True):
    """
    Detect objects in patches of an image labeled with connected components.

    Args:
        imarray (ndarray): The image array with connected components labeled.
        labels (ndarray): An array with labels for the connected components in the image.
        patch_size (int, optional): The size of patches used for object detection (default is 256).
        step_size (int, optional): The step size for moving the detection window (default is 256).
        plot (bool, optional): If True, a plot of the image with detected patches will be displayed (default is True).

    Returns:
        list: A list of dictionaries containing the results of object detection within patches.
        
    The function detects objects within patches of an image labeled with connected components. It iterates through patches of the image
    using the specified patch size and step size, counting the number of unique objects within each patch (excluding the background label).
    The results are stored in a list of dictionaries, with each dictionary containing the patch coordinates and the number of objects
    detected in that patch.

    If the plot argument is True, the function will also display a plot of the image with detected patches outlined in red rectangles.

    Example usage:
    patch_results = detect_patch_with_object(image_array, labeled_components, patch_size=256, step_size=128, plot=True)
    """
    if plot:
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(12,6))    
        # Display the binary image
        ax.imshow(imarray, cmap='gray')
    
    # Initialize a list to store the results
    patch_results = []
    
    # Iterate through patches
    for y in range(0, imarray.shape[0], step_size):
        for x in range(0, imarray.shape[1], step_size):
            # Define the patch boundaries
            patch_y1, patch_x1 = y, x
            patch_y2, patch_x2 = min(y + patch_size, imarray.shape[0]), min(x + patch_size, imarray.shape[1])
    
            # Extract the patch from the image
            patch = labels[patch_y1:patch_y2, patch_x1:patch_x2]
    
            # Find unique labels within the patch
            unique_labels = np.unique(patch)
    
            # Count the number of unique labels (excluding background label)
            num_objects_in_patch = len(unique_labels) - 1
            
            if num_objects_in_patch > 0: 
                # Store the results along with patch coordinates
                patch_results.append({
                    'patch_coords': ((patch_x1, patch_y1), (patch_x2, patch_y2)),
                    'patch_width': patch_x2 - patch_x1,
                    'patch_height': patch_y2 - patch_y1,
                    'num_objects_in_patch': num_objects_in_patch
                })
                
                if plot:
                    rect = plt.Rectangle((patch_x1, patch_y1), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
    
    if plot:
        plt.show()                
    
    return patch_results
    
    


def validate_patches(ori_img, cropped_img, lowH, lowHoffset, highH, patch_params, patch_size=256):
    
    """
    Validate and adjust patches to ensure they meet a specified size and dimensions.

    Args:
        ori_img (ndarray): The original image.
        cropped_img (ndarray): The cropped image with patches.
        lowH (int): A value representing the lower bound of height in the original image.
        lowHoffset (int): An offset value for adjusting the lower height boundary.
        highH (int): The upper bound of height in the original image.
        patch_params (list): A list of dictionaries, where each dictionary contains patch parameters, including patch height, width, and coordinates.
        patch_size (int, optional): The desired size for patches (default is 256).

    Returns:
        ndarray: The adjusted image array with validated patches.
        list: The list of dictionaries containing adjusted patch parameters.

    This function validates and adjusts patches to ensure they meet the specified 
    size and dimensions. It checks each patch's height and width, and if they 
    are smaller than the patch size, it increases the height and width to match 
    the patch_size. It also adjusts the patch coordinates accordingly. 
    The function then validates the height and width of the original image and 
    pads it if necessary to accommodate the adjusted patches.

    Example usage:
    validated_image, validated_patch_params = validate_patches(original_image, 
                cropped_image, 50, 10, 500, patch_parameters, patch_size=256)
    """
    
    h2badded = 0
    w2badded = 0
    for params in patch_params:
        if params['patch_height'] < patch_size or params['patch_width'] < patch_size:
            h2badded = patch_size - params['patch_height']
            params['patch_height'] = params['patch_height'] + h2badded
            w2badded = patch_size - params['patch_width']
            params['patch_width'] = params['patch_width'] + w2badded
            
            x1, y1 = params['patch_coords'][0]
            x2, y2 = params['patch_coords'][1]
            corH = patch_size -(y2-y1)
            corW = patch_size - (x2-x1)
            params['patch_coords'] = ((x1, y1), (x2+corW, y2+corH))
    
    ori_height = ori_img.shape[0] - lowH - lowHoffset
    cropped_height = cropped_img.shape[0] + h2badded
    
    if ori_height < cropped_height:
        ori_img = np.pad(ori_img, ((0, h2badded), (0, 0)), mode='edge')
    
    ori_width = ori_img.shape[1]
    cropped_width = cropped_img.shape[1] + w2badded
    
    if ori_width < cropped_width:
        ori_img = np.pad(ori_img, ((0, 0), (0, w2badded)), mode='edge')
        
    
    imarray = ori_img[lowH+lowHoffset:highH+h2badded, :]
    
    return imarray, patch_params
