import argparse, os, sys, shutil
import numpy as np


def search_files_extension(file_name_list, file_extensions):
    file_list = [file_name for file_name in file_name_list if file_name.endswith(file_extensions)]
    return(sorted(file_list, key=(lambda x:(len(x), x))))

def sort_slice_by_axis(image_name_list, axis_list=['z']):
    axis_case = "[" + "".join([axis_letter.lower() for axis_letter in axis_list] + [axis_letter.upper() for axis_letter in axis_list]) + "]"
    sorted_image_name_list = [image_name for image_name in image_name_list if re.search(f"[\_\-\.]{axis_case}[\_\-\.]?", image_name) is not None]
    return(sorted(sorted_image_name_list, key=(lambda x:(len(x), x))))


def search_label_files(input_directory, image_name_list):
    file_list = os.listdir(input_directory)
    image_raw_name = [image_name.split(os.extsep)[0] for image_name in image_name_list]
    label_file_list = []
    for file_name in file_list:
        if file_name.split(os.extsep)[0] in image_raw_name:
            label_file_list.append(file_name)
    return(sorted(label_file_list, key=(lambda x:(len(x), x))))


if __name__ == "__main__":
    current_filename = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(prog=current_filename, description='Split images into train/val/test directories', usage='%(prog)s [options]')
    parser.add_argument('-i', '--image_directory', dest='image_directory', required=True, help="Indicates the path of the image folder on which to run the program.")
    parser.add_argument('-l', '--label_directory', dest='label_directory', required=True, help="Indicates the path of the label folder on which to run the program.")
    parser.add_argument('-o', '--output_directory', dest='output_directory', required=True, help="Indicates the path where you want to save the dataset.")
    parser.add_argument('-a', '--axis_slice', dest='axis_slice', required=False, action="append", default=[], help="Indicates the slicing axis along which we want to build our dataset.")
    parser.add_argument('-x', '--image_extension', dest='image_extension', required=False, action="append", default=['png', 'jpg', 'tif', 'tiff'], help="Indicates the extension of images you want to search for.")
    parser.add_argument('-n', '--n_images', dest='n_images', type=int, required=False, help="Indicates the number of images you want in your dataset directory.")
    parser.add_argument('-r', '--random_split', dest='random_split', action="store_true", default=False, help="Indicates random splitting of data (if not, the data will be split into 3 successive parts to create training, validation and test datasets.")
    parser.add_argument('-s', '--seed', dest='seed', type=int, required=False, help="Indicates the seed value for reproducible image selection, in case of random split only.")
    parser.add_argument('-p', '--part_index', dest='part_index', type=int, choices=[0, 1, 2], nargs='*', action="store", default=[], help="List of indices (if random_split is False) corresponding to the order in which data will be separated and arranged in training (index 0), validation (index 1) and test (index 2) datasets.")
    parser.add_argument('--train_data_ratio', dest='train_data_ratio', type=float, required=False, default=0.70, help="Indicates the ratio of images you want in your train dataset (default 70%).")
    parser.add_argument('--val_data_ratio', dest='val_data_ratio', type=float, required=False, default=0.10, help="Indicates the ratio of images you want in your validation dataset (default 10%).")
    parser.add_argument('--test_data_ratio', dest='test_data_ratio', type=float, required=False, default=0.20, help="Indicates the ratio of images you want in your test dataset (default 20%).")
    args = parser.parse_args()
    image_directory = os.path.normpath(args.image_directory)
    label_directory = os.path.normpath(args.label_directory)
    output_directory = os.path.normpath(args.output_directory)
    image_extension_list = args.image_extension
    random_split = args.random_split
    dataset_index_dict = {0:"train", 1:"validation", 2:"test"}
    split_method = "Random split"*int(random_split) + "Ordered split"*(1-int(random_split))
    if random_split:
        dataset_index_order = [0, 1, 2]
    else:
        dataset_index_order = [int(i) for i in args.part_index]
        dataset_index_order += [i for i in range(3) if i not in dataset_index_order]
    dataset_name_order = [dataset_index_dict[i] for i in dataset_index_order]
    axis_slice_list = list(np.unique(args.axis_slice))
    axis_slice_list += ['z']*(int(len(axis_slice_list) == 0))
    original_image_list = search_files_extension(os.listdir(image_directory), tuple(image_extension_list))
    n_original_images = len(original_image_list)
    image_name_list = sort_slice_by_axis(original_image_list, axis_list=axis_slice_list)
    label_name_list = search_label_files(label_directory, image_name_list)
    train_ratio = max(0.0, min(1.0, args.train_data_ratio))
    val_ratio = max(0.0, min(1.0-train_ratio, args.val_data_ratio))
    test_ratio = max(0.0, min(1.0-(train_ratio+val_ratio), args.test_data_ratio))
    n_images = len(image_name_list)
    if args.n_images is None:
        n_data = len(image_name_list)
    else:
        n_data = min(n_images, args.n_images)
    if args.seed is not None:
        seed = args.seed
        np.random.seed(seed)
    else:
        seed = "no seed"
    n_train_data = round(train_ratio * n_data)
    n_val_data = round(val_ratio * n_data)
    n_test_data = round(test_ratio * n_data)
    # Create the train/val/test dataset
    output_image_path = os.path.join(output_directory, "images")
    output_image_train_path = os.path.join(output_image_path, "train")
    output_image_val_path = os.path.join(output_image_path, "val")
    output_image_test_path = os.path.join(output_image_path, "test")
    output_label_path = os.path.join(output_directory, "labels")
    output_label_train_path = os.path.join(output_label_path, "train")
    output_label_val_path = os.path.join(output_label_path, "val")
    output_label_test_path = os.path.join(output_label_path, "test")
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)
    os.makedirs(output_image_train_path, exist_ok=True)
    os.makedirs(output_image_val_path, exist_ok=True)
    os.makedirs(output_image_test_path, exist_ok=True)
    os.makedirs(output_label_train_path, exist_ok=True)
    os.makedirs(output_label_val_path, exist_ok=True)
    os.makedirs(output_label_test_path, exist_ok=True)
    # Split images into train/val/test/val
    if random_split:
        idx_train_data = np.random.choice(np.arange(n_images), n_train_data, replace=False)
        remaining_index = np.delete(np.arange(n_images), idx_train_data)
        idx_val_choice = np.random.choice(np.arange(remaining_index.size), n_val_data, replace=False)
        idx_val_data = remaining_index[idx_val_choice]
        remaining_index = np.delete(remaining_index, idx_val_choice)
        idx_test_choice = np.random.choice(np.arange(remaining_index.size), n_test_data, replace=False)
        idx_test_data = remaining_index[idx_test_choice]
    else:
        n_sum_data = 0
        for i in dataset_index_order:
            if (i == 0):
                idx_train_data = np.arange(n_sum_data, n_sum_data + n_train_data)
                n_sum_data += n_train_data
            elif (i == 1):
                idx_val_data = np.arange(n_sum_data, n_sum_data + n_val_data)
                n_sum_data += n_val_data
            else:
                idx_test_data = np.arange(n_sum_data, n_sum_data + n_test_data)
                n_sum_data += n_test_data
    print(f"Image search result : {n_images} found from {n_original_images}, {n_data} will be copied into the dataset")
    print("Image search parameters :")
    print(f"\t- Image extensions : {(', ').join(image_extension_list)}")
    print(f"\t- Slice axis : {(', ').join([axis_letter.upper() for axis_letter in axis_slice_list])}")
    print("\nSplit data parameters :")
    print(f"\t- Split method : {split_method}")
    if not random_split:
        print(f"\t- Dataset split order : {(", ").join(dataset_name_order)}")
    print(f"\t- Train {train_ratio:2.2%} : {n_train_data} images")
    print(f"\t- Validation {val_ratio:2.2%} : {n_val_data} images")
    print(f"\t- Test {test_ratio:2.2%} : {n_test_data} images")
    print(f"\nImage directory : \'{image_directory}\'")
    print(f"Label directory : \'{label_directory}\'")
    print(f"Dataset directory : \'{output_directory}\'")
    print(f"Seed value : {seed}")
    for i in dataset_index_order:
        if (i == 0):
            for idx_train in idx_train_data:
                image_name = image_name_list[idx_train]
                label_name = label_name_list[idx_train]
                input_image_path = os.path.join(image_directory, image_name)
                input_label_path = os.path.join(label_directory, label_name)
                output_image_path = os.path.join(output_image_train_path, image_name)
                output_label_path = os.path.join(output_label_train_path, label_name)
                shutil.copy2(input_image_path, output_image_path)
                shutil.copy2(input_label_path, output_label_path)
        elif (i == 1):
            for idx_val in idx_val_data:
                image_name = image_name_list[idx_val]
                label_name = label_name_list[idx_val]
                input_image_path = os.path.join(image_directory, image_name)
                input_label_path = os.path.join(label_directory, label_name)
                output_image_path = os.path.join(output_image_val_path, image_name)
                output_label_path = os.path.join(output_label_val_path, label_name)
                shutil.copy2(input_image_path, output_image_path)
                shutil.copy2(input_label_path, output_label_path)
        else:
            for idx_test in idx_test_data:
                image_name = image_name_list[idx_test]
                label_name = label_name_list[idx_test]
                input_image_path = os.path.join(image_directory, image_name)
                input_label_path = os.path.join(label_directory, label_name)
                output_image_path = os.path.join(output_image_test_path, image_name)
                output_label_path = os.path.join(output_label_test_path, label_name)
                shutil.copy2(input_image_path, output_image_path)
                shutil.copy2(input_label_path, output_label_path)
