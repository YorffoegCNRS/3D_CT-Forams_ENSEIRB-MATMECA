import argparse, cv2, os, torch
import math as m
import numpy as np
import numpy.linalg as lg
import tifffile as tiff
import pandas as pd

from ultralytics import SAM
from tqdm import tqdm



def draw_image_masks(image, bounding_boxes, bgr_color=(0,0,0)):
    for bounding_box in bounding_boxes:
        a_min, b_min, a_max, b_max = bounding_box
        cv2.rectangle(image, (a_min, b_min), (a_max, b_max), color=tuple(bgr_color), thickness=-1)
    return(image)



def apply_sam_model(image, bounding_box, sam_model):
    bboxes = torch.tensor([bounding_box])  # Convert to tensor
    # Normalize the image
    image = image.astype(np.float32) / 255.0
    # Ensure the image has three channels
    if image.ndim == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    # Resize the image to a compatible size (e.g., 1024x1024)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (1024, 1024))
    scale_x = 1024 / original_size[1]
    scale_y = 1024 / original_size[0]
    # Adjust bounding box coordinates to match the resized image
    bbox_resized = [
        int(bounding_box[0] * scale_x),  # a_min
        int(bounding_box[1] * scale_y),  # b_min
        int(bounding_box[2] * scale_x),  # a_max
        int(bounding_box[3] * scale_y)   # b_max
    ]
    # Convert to tensor for inference
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0)  # BCHW
    # Perform segmentation using the SAM model
    results = sam_model(image_tensor, bboxes=torch.tensor([bbox_resized]))
    # Collect masks from results
    if results and hasattr(results[0], 'masks'):
        mask = results[0].masks.data  # Shape: (1, height, width)
        # Convert the boolean mask to uint8 (0 or 255)
        mask = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to numpy and scale to 255
        # Resize the mask back to the original image dimensions
        mask_resized = cv2.resize(mask[0], (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask_resized  # Return the resized mask
    return None



def rectangle_intersection(bbox_1, bbox_2):
    if ((((bbox_2[0,0] <= bbox_1[0,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[0,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[0,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[0,1] <= bbox_1[1,1]))) or
        (((bbox_2[0,0] <= bbox_1[0,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[1,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[0,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[1,1] <= bbox_1[1,1]))) or
        (((bbox_2[0,0] <= bbox_1[1,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[1,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[1,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[1,1] <= bbox_1[1,1]))) or
        (((bbox_2[0,0] <= bbox_1[1,0] <= bbox_2[1,0]) and (bbox_2[0,1] <= bbox_1[0,1] <= bbox_2[1,1])) or ((bbox_1[0,0] <= bbox_2[1,0] <= bbox_1[1,0]) and (bbox_1[0,1] <= bbox_2[0,1] <= bbox_1[1,1])))):
        return(True)
    return(False)



def compute_max_point_frames(point_coords, box_width, n_frames, n_mask_frames, tif_shape, mask_direction='z'):
    direction_dict = {'x':0, 'y':1, 'z':2}
    mask_axis = direction_dict[mask_direction.lower()]
    other_axis = np.delete(np.arange(3), mask_axis)
    half_box_width = box_width // 2
    n_min_frames = n_frames // 2 + 1
    n_max_frames = n_frames // 2 + n_mask_frames // 2
    n_points = point_coords.shape[0]
    direction_shape = [0, tif_shape[2-mask_axis]]
    min_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    min_mask_frame_array[:, 0] = point_coords[:, mask_axis] - n_max_frames
    min_mask_frame_array[:, 1] = point_coords[:, mask_axis] - n_min_frames
    max_mask_frame_array = np.zeros([n_points, 2], dtype='int32')
    max_mask_frame_array[:, 0] = point_coords[:, mask_axis] + n_min_frames
    max_mask_frame_array[:, 1] = point_coords[:, mask_axis] + n_max_frames
    for i in range(n_points):
        for j in range(i+1, n_points):
            frame_interval_0 = np.array([point_coords[i, mask_axis] - n_frames // 2, point_coords[i, mask_axis] + n_frames // 2])
            frame_interval_1 = np.array([point_coords[j, mask_axis] - n_frames // 2, point_coords[j, mask_axis] + n_frames // 2])
            box_coords_0 = np.array([[point_coords[i, other_axis[0]] - half_box_width, point_coords[i, other_axis[1]] - half_box_width],
                                     [point_coords[i, other_axis[0]] + half_box_width, point_coords[i, other_axis[1]] + half_box_width]])
            box_coords_1 = np.array([[point_coords[j, other_axis[0]] - half_box_width, point_coords[j, other_axis[1]] - half_box_width],
                                     [point_coords[j, other_axis[0]] + half_box_width, point_coords[j, other_axis[1]] + half_box_width]])
            if rectangle_intersection(box_coords_0, box_coords_1):
                if (min_mask_frame_array[i, 0] <= frame_interval_1[0] <= min_mask_frame_array[i, 1]):
                    min_mask_frame_array[i, :] = np.array([max(min_mask_frame_array[i, 0], min_mask_frame_array[i, 0]), min(min_mask_frame_array[i, 1], frame_interval_1[0] - 1)], dtype='int32')
                if (min_mask_frame_array[j, 0] <= frame_interval_0[0] <= min_mask_frame_array[j, 1]):
                    min_mask_frame_array[j, :] = np.array([max(min_mask_frame_array[j, 0], min_mask_frame_array[j, 0]), min(min_mask_frame_array[j, 1], frame_interval_0[0] - 1)], dtype='int32')
                if (max_mask_frame_array[i, 0] <= frame_interval_1[0] <= max_mask_frame_array[i, 1]):
                    max_mask_frame_array[i, :] = np.array([max(max_mask_frame_array[i, 0], max_mask_frame_array[i, 0]), min(max_mask_frame_array[i, 1], frame_interval_1[0] - 1)], dtype='int32')
                    min_mask_frame_array[j, :] = np.array([max(min_mask_frame_array[j, 0], min_mask_frame_array[j, 0]), min(min_mask_frame_array[j, 1], frame_interval_1[0] - 1)], dtype='int32')
                if (max_mask_frame_array[j, 0] <= frame_interval_0[0] <= max_mask_frame_array[j, 1]):
                    max_mask_frame_array[j, :] = np.array([max(max_mask_frame_array[j, 0], max_mask_frame_array[j, 0]), min(max_mask_frame_array[j, 1], frame_interval_0[0] - 1)], dtype='int32')
                    min_mask_frame_array[i, :] = np.array([max(min_mask_frame_array[i, 0], min_mask_frame_array[i, 0]), min(min_mask_frame_array[i, 1], frame_interval_0[0] - 1)], dtype='int32')
                if (min_mask_frame_array[i, 0] <= frame_interval_1[1] <= min_mask_frame_array[i, 1]):
                    min_mask_frame_array[i, :] = np.array([max(min_mask_frame_array[i, 0], frame_interval_1[1] + 1), min(min_mask_frame_array[i, 1], min_mask_frame_array[i, 1])], dtype='int32')
                    max_mask_frame_array[j, :] = np.array([max(max_mask_frame_array[j, 0], frame_interval_1[1] + 1), min(max_mask_frame_array[j, 1], min_mask_frame_array[i, 1])], dtype='int32')
                if (min_mask_frame_array[j, 0] <= frame_interval_0[1] <= min_mask_frame_array[j, 1]):
                    min_mask_frame_array[j, :] = np.array([max(min_mask_frame_array[j, 0], frame_interval_0[1] + 1), min(min_mask_frame_array[j, 1], min_mask_frame_array[j, 1])], dtype='int32')
                    max_mask_frame_array[i, :] = np.array([max(max_mask_frame_array[i, 0], frame_interval_0[1] + 1), min(max_mask_frame_array[i, 1], min_mask_frame_array[j, 1])], dtype='int32')
                if (max_mask_frame_array[i, 0] <= frame_interval_1[1] <= max_mask_frame_array[i, 1]):
                    max_mask_frame_array[i, :] = np.array([max(max_mask_frame_array[i, 0], frame_interval_1[1] + 1), min(max_mask_frame_array[i, 1], max_mask_frame_array[i, 1])], dtype='int32')
                if (max_mask_frame_array[j, 0] <= frame_interval_0[1] <= max_mask_frame_array[j, 1]):
                    max_mask_frame_array[j, :] = np.array([max(max_mask_frame_array[j, 0], frame_interval_0[1] + 1), min(max_mask_frame_array[j, 1], max_mask_frame_array[j, 1])], dtype='int32')
    return(min_mask_frame_array, max_mask_frame_array)



def save_images_and_annotations(tif_path, csv_path, output_folder, box_width, num_frames, axis_indices, model_path, base_image_name, apply_sam, num_frames_to_mask, gray_value=128, csv_separator=';'):
    direction_dict = {0:'z', 1:'y', 2:'x'}
    draw_mask = bool(num_frames_to_mask > 1)
    bgr_color = (gray_value, gray_value, gray_value)
    
    if apply_sam:
        sam_model = SAM(model_path)
    
    # Create the output folder and necessary subfolders
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    labels_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    
    # Load the TIFF file
    tif_data = tiff.imread(tif_path)
    tif_shape = tif_data.shape[:3]
    # Load CSV data
    df = pd.read_csv(csv_path, delimiter=csv_separator)
    
    points = df.to_numpy()[:, 1:]
    n_points = points.shape[0]
    # For each direcion
    for direction in axis_indices: # 0: z, 1: y, 2: x
        d = direction_dict[direction]
        
        if draw_mask:
            min_masked_frames, max_masked_frames = compute_max_point_frames(df.to_numpy()[:, 1:], box_width, num_frames, num_frames_to_mask, tif_shape, mask_direction=d)
        
        label_frame = np.zeros([n_points, 2], dtype='int32')
        label_frame[:, 0] = points[:, 2-direction] - num_frames // 2
        label_frame[:, 1] = points[:, 2-direction] + num_frames // 2
        
        print(f"Current direction : axis {d.upper()}")
        # Iterate through each image in the TIFF file
        ax_point = np.delete(points, 2-direction, axis=1)
        bbox_array = np.zeros([n_points, 4], dtype='int32')
        bbox_array[:, 0] = np.maximum(np.zeros(n_points), ax_point[:, 0] - box_width // 2)
        bbox_array[:, 1] = np.maximum(np.zeros(n_points), ax_point[:, 1] - box_width // 2)
        bbox_array[:, 2] = np.minimum(np.full([n_points], tif_shape[1]), ax_point[:, 0] + box_width // 2)
        bbox_array[:, 3] = np.minimum(np.full([n_points], tif_shape[2]), ax_point[:, 1] + box_width // 2)
        for idx in tqdm(range(tif_data.shape[direction]), desc="Processing images"):
            # Extract the slice along the specified direction
            img_array = np.take(tif_data, indices=idx, axis=direction).copy()
            
            # Convert to BGR format if necessary
            if img_array.ndim == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            idx_annotation = np.where(np.logical_and(label_frame[:, 0] <= idx, idx <= label_frame[:, 1]))[0]
            
            # Gather bounding boxes for the current index and the surrounding frames in that direction
            bounding_boxes = []
            
            axis_point = points[idx_annotation, :]
            axis_point = np.delete(axis_point, 2-direction, axis=1)
            for idx_row in range(axis_point.shape[0]):
                a_min = max(axis_point[idx_row, 0] - box_width // 2, 0)
                a_max = min(axis_point[idx_row, 0] + box_width // 2, img_array.shape[1])
                b_min = max(axis_point[idx_row, 1] - box_width // 2, 0)
                b_max = min(axis_point[idx_row, 1] + box_width // 2, img_array.shape[0])
                bounding_boxes.append([a_min, b_min, a_max, b_max])
            
            # Initialize a list for storing annotations
            annotations, bounding_boxes_list = [], []
            
            # Perform inference for each bounding box
            for bbox in bounding_boxes:
                if not apply_sam:
                    # Calculate YOLO format bounding box
                    x_center = (bbox[0] + bbox[2]) / (2 * img_array.shape[1])
                    y_center = (bbox[1] + bbox[3]) / (2 * img_array.shape[0])
                    width = (bbox[2] - bbox[0]) / img_array.shape[1]
                    height = (bbox[3] - bbox[1]) / img_array.shape[0]
                    annotations.append(f"0 {x_center} {y_center} {width} {height}")
                    bounding_boxes_list = bounding_boxes
                else:
                    mask = apply_sam_model(img_array, bbox, sam_model)
                    if mask is not None:
                        y_indices, x_indices = np.where(mask)  # Get the coordinates of the mask
                        if len(y_indices) > 0 and len(x_indices) > 0:  # Ensure there are coordinates
                            bbox = [
                                np.min(x_indices),  # Left
                                np.min(y_indices),  # Top
                                np.max(x_indices),  # Right
                                np.max(y_indices)   # Bottom
                            ]
                        # Calculate YOLO format bounding box
                        x_center = (bbox[0] + bbox[2]) / (2 * img_array.shape[1])
                        y_center = (bbox[1] + bbox[3]) / (2 * img_array.shape[0])
                        width = (bbox[2] - bbox[0]) / img_array.shape[1]
                        height = (bbox[3] - bbox[1]) / img_array.shape[0]
                        annotations.append(f"0 {x_center} {y_center} {width} {height}")
                        bounding_boxes_list.append(bbox)
            
            # Save the image untouched in the output folder if it has not been processed before
            output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
            if not os.path.exists(output_image_path):
               cv2.imwrite(output_image_path, img_array)
            # Write annotations to a file with the same name as the image in the labels folder
            annotation_file_path = os.path.join(labels_folder, f'{base_image_name}_{d}_{idx}.txt')
            with open(annotation_file_path, 'w') as ann_file:
                for ann in annotations:
                    ann_file.write(f"{ann}\n")
        # If necessary, draw black masks on the images
        if (draw_mask == True):
            print(f"Direction : axis {d.upper()}")
            for idx in tqdm(range(tif_data.shape[direction]), desc="Drawing mask"):
                idx_masked_points = np.where(np.logical_or(np.logical_and(min_masked_frames[:, 0] <= idx, idx <= min_masked_frames[:, 1]), np.logical_and(max_masked_frames[:, 0] <= idx, idx <= max_masked_frames[:, 1])))[0]
                mask_bbox = list(bbox_array[idx_masked_points, :])
                if (0 <= idx < tif_data.shape[direction]):  # Check the boundaries
                    # get the image from the images_folder if already processed before
                    if os.path.exists(os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')):
                        img_to_process = cv2.imread(os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png'))
                        img_to_process = draw_image_masks(img_to_process, mask_bbox, bgr_color)
                        # Réenregistrement de l'image modifiée
                        output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
                        cv2.imwrite(output_image_path, img_to_process)
                    else:
                        # Extract the slice along the specified direction
                        img_to_process = np.take(tif_data, indices=idx, axis=direction).copy()
                        # Convert to BGR format if necessary
                        if img_to_process.ndim == 2:  # Grayscale
                            img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
                        img_to_process = draw_image_masks(img_to_process, mask_bbox, bgr_color)
                        # Enregistrement de l'image modifiée
                        output_image_path = os.path.join(images_folder, f'{base_image_name}_{d}_{idx}.png')
                        cv2.imwrite(output_image_path, img_to_process)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TIFF images, apply segmentation using a SAM model, and save images along with LabelMe annotations.')
    parser.add_argument('--tif_path', type=str, help='Path to the TIFF file.')
    parser.add_argument('--csv_path', type=str, help='Path to the input CSV file with coordinates.')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder.')
    parser.add_argument('--box_width', default=20, type=int, help='Width of the bounding box around each point.')
    parser.add_argument('--num_frames', default=5, type=int, help='Total number of frames (above and below in total) to consider.')
    parser.add_argument('--axis', required=False, action="append", default=[], help="Indicates the slicing axis along which we want to build our dataset (all axis by default).")
    parser.add_argument('--model_path', default="mobile_sam.pt",type=str, help='Path to the SAM model file.')
    parser.add_argument('--base_image_name', default="image", type=str, help='Base name for output images and annotations.')
    parser.add_argument('--apply_sam', action="store_true", help="Use SAM to refine the bbox")
    parser.add_argument('--num_frames_to_mask', default=8, type=int, help='Total number of frames (above and below in total) to consider.')
    parser.add_argument('--mask_color', default=128, type=int, help="Grayscale value used as mask color (default 128)")
    parser.add_argument('--csv_sep', default=';', type=str, help="CSV delimiter character (default \';\')")
    args = parser.parse_args()
    direction_dict = {'z':0, 'y':1, 'x':2}
    axis_name_list = args.axis + ['z', 'y', 'x']*int(len(args.axis) == 0)
    axis_index_list = [direction_dict[axis_name] for axis_name in axis_name_list if axis_name in direction_dict.keys()]
    save_images_and_annotations(args.tif_path, args.csv_path, args.output_folder, args.box_width, args.num_frames, axis_index_list, args.model_path, args.base_image_name, args.apply_sam, args.num_frames_to_mask, args.mask_color, args.csv_sep)
    print(f'Images and annotations saved in {args.output_folder}.')
