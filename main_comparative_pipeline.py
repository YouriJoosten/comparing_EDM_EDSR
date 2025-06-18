import os
import numpy as np
import pandas as pd
import torch
import cv2
import sys
import time
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as functional
from skimage.metrics import structural_similarity as ssim

print(sys.prefix)
if "bachelor" not in sys.prefix:
    print("Not in correct environment")

def load_data_with_labels(images_path, labels_path):
    """
    Loads the image paths with their corresponding label into a list.
    :param images_path:
    :param labels_path:
    :return:
    """
    df = pd.read_csv(labels_path)
    data=[]
    for i, row in df.iterrows():
        image_path = f"{images_path}/{row['Path']}"
        label = row['ClassId']
        data.append((image_path, label))
    return data


def add_jpeg_compression(image, quality):
    """
    Adds JPEG compression to the image, this function is used in degrading the images.
    :param image:
    :param quality:
    :return:
    """
    encoding = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, jpeg_image = cv2.imencode('.jpg',image, encoding)
    return  cv2.imdecode(jpeg_image, cv2.IMREAD_COLOR)


def add_poisson_noise(image, poisson_intensity):
    """
    Adds poisson noise to the image, this function is used in degrading the images.
    :param image:
    :param poisson_intensity:
    :return:
    """
    normalized_image = image.astype(np.float32) / 255.0
    generate_poisson_noise = np.random.poisson(poisson_intensity, size=normalized_image.shape) / 255.0
    image_with_poisson_noise = normalized_image + generate_poisson_noise  # adding poisson noise to the image
    formatted_image = (np.clip(image_with_poisson_noise , 0, 1)*255).astype(np.uint8)
    return formatted_image

def resize_image(image_path, size):
    """
    Resizes the image, this function is used in degrading the images.
    :param image_path:
    :param size:
    :return:
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.resize(img, dsize = size )

def degrade_image(image_path,size):
    """
    Degrades the images by resizing it, adding poisson noise and using JPEG compression.
    :param image_path:
    :param size:
    """
    resized_image = resize_image(image_path, size)
    noisy_image = add_poisson_noise(resized_image, 1)
    compressed_and_noisy_image = add_jpeg_compression(noisy_image,quality=95)
    return compressed_and_noisy_image

def degrade_the_data(image_size,training_data,train_dir, database_root):
    """
    This functions performs the data degradation and saves the degraded images to a new directory.
    :param image_size:
    :param training_data:
    :param train_dir:
    :param database_root:
    :return:
    """
    degraded_training_data_path_and_label = []
    start_time = time.time()
    for image_path,label in training_data:
        # create subdirectories
         label_dir = os.path.join(train_dir, str(label))
         os.makedirs(label_dir, exist_ok=True)
         # degrade the image and save it to the subdirectory corresponding to the class
         degraded_image = degrade_image(image_path, image_size)
         output_path = os.path.join(label_dir, os.path.basename(image_path))
         cv2.imwrite(output_path, degraded_image)
         degraded_training_data_path_and_label.append((output_path, image_path))
    # create a csv file with the paths and labels of the degraded images for training the models.
    degraded_database = pd.DataFrame(degraded_training_data_path_and_label, columns=["Path", "ClassId"])
    degraded_database.to_csv(f"{database_root}/Train_degraded_{image_size[0]}x{image_size[1]}.csv", index=False)
    end_time = time.time()
    print(f"Execution time preparing training data: {end_time - start_time} seconds")

def flatten_image_directory(input_dir, output_dir):
    """
    Flattens a directory where the images are placed in subdirectories by copying all the files to a new directory.
    :param input_dir:
    :param output_dir:
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for subdirectory, _, images in os.walk(input_dir):
        for image in images:
            if image.lower().endswith(('.png')):
                original_path = os.path.join(subdirectory, image)
                new_path = os.path.join(output_dir, image)
                # copy the file to a flat directory
                shutil.copy2(original_path, new_path)
                count += 1
    print(f"Copied {count} images to flat directory: {output_dir}")

def resize_clean_images_into_directory(flat_dir):
    # Load the CSV file
    image_path = 'Path'
    output_dir = "C:/Users/youri/BachelorThesisLocal/database/resized_test"
    df = pd.read_csv("C:/Users/youri/BachelorThesisLocal/database/Test.csv")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        image = os.path.basename(row[image_path])
        src_path = os.path.join(flat_dir, image)

        if not os.path.exists(src_path):
            print(f"File not found: {src_path}")
            continue

        try:
            resized_img = resize_image(src_path, (32, 32))
            destination_path = os.path.join(output_dir, image)
            cv2.imwrite(destination_path, resized_img)

        except Exception as e:
            print(f"Error processing {image}: {e}")

def prepare_edm_training_data(image_size,input_root,output_root, csv_output):
    training_image_paths_and_labels = []
    start_time = time.time()

    for label in os.listdir(input_root):
        label_path = os.path.join(input_root, label)

        # make sure output label directory exists
        output_label_dir = os.path.join(output_root, label)
        os.makedirs(output_label_dir, exist_ok=True)
        print("processing label", label)

        for file in os.listdir(label_path):
            input_image_path = os.path.join(label_path, file)
            output_image_path = os.path.join(output_label_dir, file)

            # Resize and save
            image = resize_image(input_image_path, image_size)
            cv2.imwrite(output_image_path, image)

            # Store relative path and label
            relative_path = os.path.relpath(output_image_path, output_root)
            training_image_paths_and_labels.append((relative_path, int(label)))

    # Save CSV
    df = pd.DataFrame(training_image_paths_and_labels, columns=["Path", "ClassId"])
    df.to_csv(csv_output, index=False)

    end_time = time.time()
    print(f"Execution time preparing training data: {end_time - start_time} seconds")


def prepare_testing_data(image_size, testing_data, test_dir, database_root):
    resized_test_data_path_and_label = []
    start_time = time.time()
    for image_path, label in testing_data:
        # create subdirectories
        label_dir = os.path.join(test_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        # resize the image and save it to the directory for the given label
        resized_image = resize_image(image_path, image_size)
        output_path = os.path.join(label_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, resized_image)
        resized_test_data_path_and_label.append((output_path, image_path))
    # create a csv file with the paths and labels of the resized images for training the model later on in the pipeline function above.
    resized_database = pd.DataFrame(resized_test_data_path_and_label, columns=["Path", "ClassId"])
    resized_database.to_csv(f"{database_root}/Test_degraded_{image_size[0]}x{image_size[1]}.csv", index=False)
    end_time = time.time()
    print(f"Execution time preparing testing data: {end_time - start_time} seconds")

def prepare_dataset(image_size):
    database_root = "C:/Users/youri/BachelorThesisLocal/database"
    train_data = load_data_with_labels(images_path=database_root, labels_path=f"{database_root}/Train.csv", loading_traindata=True)
    test_data = load_data_with_labels(images_path="database/", labels_path="database/Test.csv", loading_traindata=False)
    print("length of train data", len(train_data))
    print("length of test data", len(test_data), "\n")
    print("train data element: ", train_data[0])
    print("test data element: ", test_data[0], "\n")
    test_dir = f"{database_root}/Test_degraded_{image_size[0]}x{image_size[1]}"
    train_dir = f"{database_root}/Train_degraded_{image_size[0]}x{image_size[1]}"
    prepare_edm_training_data(image_size, train_data, train_dir, database_root)
    prepare_testing_data(image_size, test_data, test_dir, database_root)

def load_YOLO_data(image_size, source_img_dir, dest_img_dir, path_to_csv):
    df = pd.read_csv(path_to_csv)
    print(dest_img_dir)
    for _, row in df.iterrows():
        filename = row['Path']
        label = str(row['ClassId']).zfill(2)  # formats the directory names

        src_path = os.path.join(source_img_dir, os.path.basename(filename))
        destination_folder = os.path.join(dest_img_dir, label)
        destination_path = os.path.join(destination_folder, os.path.basename(filename))

        # Make class folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        # Copy image into subirectory folder
        print("Source Path ", src_path,image_size)
        img = resize_image(src_path,image_size)
        print("Destination Path ", destination_path)
        cv2.imwrite(destination_path, img)


# Compute PSNR
def compute_psnr(img1, img2):
    mse = functional.mse_loss(img1, img2)
    if mse == 0:
        return float(100)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_ssim(img1, img2):
    img1_np = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=-1)

transformer = transforms.Compose([transforms.ToTensor()])

def evaluate_edsr_outputs(edsr_output_dir, clean_dir):
    psnr_ssim_scores = []
    save_path = 'edsr_metrics.xlsx'
    device = 'cuda'

    for subdirectory, _, images in os.walk(edsr_output_dir):
        for image_path in images:
            if not image_path.endswith('.png'):
                continue

            denoised_path = os.path.join(subdirectory, image_path)
            clean_filename = image_path.replace('_x1_SR', '')
            clean_path = os.path.normpath(os.path.join(clean_dir, clean_filename))

            if not os.path.exists(clean_path):
                print(f"Clean image not found: {clean_path}")
                continue

            denoised_img = Image.open(denoised_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')

            denoised_tensor = transformer(denoised_img).unsqueeze(0).to(device)
            clean_tensor = transformer(clean_img).unsqueeze(0).to(device)
            psnr_val = compute_psnr(clean_tensor, denoised_tensor)
            ssim_val = compute_ssim(clean_tensor, denoised_tensor)
            psnr_ssim_scores.append({'Filename': image_path, 'PSNR': psnr_val.item(), 'SSIM': ssim_val})
    # Save metrics to CSV and Excel files
    df = pd.DataFrame(psnr_ssim_scores)
    df.to_csv(save_path.replace('.xlsx', '.csv'), index=False)
    df.to_excel(save_path, index=False)
    print(f"Saved PSNR/SSIM results to: {save_path}")

def plot_diffusion_accuracy_per_step():
    """
    Plots how the accuracy evolves when the number of diffusion steps increases.
    :return:
    """
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracies = []

    plt.figure(figsize=(8, 5))
    plt.plot(steps, accuracies, marker='o', linestyle='-')
    plt.title('Effect of diffusion Steps on Classification Accuracy')
    plt.xlabel('Number of diffusion steps')
    plt.ylabel('Classification accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("diffusion_steps_vs_accuracy.png")
    plt.show()

def plot_classifiers_accuracy():
    """
    Plots the accuracies of the classifiers.
    :return:
    """
    methods_denoised = ['Base YOLO Classifier', 'EDSR denoised', 'EDM denoised']
    methods = ['Clean images', 'EDSR Upsampled', 'EDM Upsampled']
    accuracies_using_baseline = [0.954, 0.0589, 0.0730]
    accuracies_upsampled = [0.954, 0.962, 0.0478]  # Example values

    plt.figure(figsize=(7, 5))
    plt.bar(methods, accuracies_using_baseline, color=['blue', 'red', 'green'])
    plt.title('Classifier Accuracy Comparison on different validation type images')
    plt.ylabel('Top 1 Accuracy')
    plt.ylim(0, 1.0)
    for i, acc in enumerate(accuracies_using_baseline):
        plt.text(i, acc, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig("base_classifier_comparison.png")
    plt.show()


def print_dataset_distribution(data_dir):
    """
    Plots the distribution of the data over the classes.
    :param data_dir:
    :return:
    """
    class_counts = {}

    # List subdirectories (classes)
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = 0
            for image in os.listdir(class_path): # for each image in the subdirectory
                image_path = os.path.join(class_path, image)
                if os.path.isfile(image_path): #counts only the files
                    count += 1
            class_counts[class_name] = count

    # Sort classes numerically, as otherwise they are not shown in numerical order
    class_index = list(class_counts.keys())
    class_keys = []
    counts = []
    for index in class_index:
        class_keys.append(int(index))
    class_keys.sort()
    for key in class_keys:
        counts.append(class_counts[str(key)])

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(class_keys, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    print_dataset_distribution("C:/Users/youri/BachelorThesisLocal/database/YOLO_training_EDSR_finetuning_data/train")
    evaluate_edsr_outputs("C:/Users/youri/BachelorThesisLocal/denoised/test1/edsr/upsampled_x1_classification_dir/val","C:/Users/youri/BachelorThesisLocal/database/Test_degraded_flattened/HR"  )

