import os
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import dnnlib
from generate import edm_sampler
import csv
import time
from skimage.metrics import structural_similarity as ssim

# denoising parameters
batch_size = 128
resolution = (32,32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paths
input_dir_base = 'C:/Users/youri/BachelorThesisLocal/database/test_degraded_32x32_images'
clean_dir = 'C:/Users/youri/BachelorThesisLocal/denoised/test1/edm/edm/clean'
network_pkl = 'Models/00005-edm_traffic_sign_32x32/network-snapshot-010000.pkl'
output_base_dir = 'C:/Users/youri/BachelorThesisLocal/denoised/test1/edm/edm/upsampled_out'


def transformer():
    """
    This function makes sure all images are 32x32 and then transforms the image to a Pytorch tensor (needed by EDM).
    It makes use of the torchvision library.
    :return:
    """
    return transforms.Compose([transforms.Resize(resolution),transforms.ToTensor()])

    # ---------------------- Dataset ---------------------- #
class ImageFolderDataset(Dataset):
    """
    Implementation of the dataloader class from torch.utils.Dataloader in order to use batch processing for EDM denoising.
    """
    def __init__(self, root, transform=None):
        """
        Loads in the dataset paths.
        :param root:
        :param transform:
        """
        self.image_paths = []
        for class_folder in os.listdir(root):
            class_path = os.path.join(root, class_folder)
            if os.path.isdir(class_path):
                for image in os.listdir(class_path):
                    if image.lower().endswith(('.png')):
                        self.image_paths.append(os.path.join(class_path, image))
        self.transform = transformer()

    def __len__(self):
        """
        Returns the length of the dataset
        :return:
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Returns the image as a pytorch tensor
        :param index:
        :return:
        """
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path


def compute_psnr(predicted, target):
    """
    Computes the PSNR score

    :param predicted:
    :param target:
    :return:
    """
    mse = torch.mean((predicted - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0, device=predicted.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(predicted, target):
    """
    Computes the SSIM score using the skimage.metrics library. The images are first formatted correctly for the SSIM function.

    :param predicted:
    :param target:
    :return:
    """
    pred_np = predicted.squeeze().permute(1, 2, 0).cpu().numpy()
    target_np = target.squeeze().permute(1, 2, 0).cpu().numpy()
    return ssim(pred_np, target_np, data_range=1.0, channel_axis=-1)

def save_metrics(psnr_noisy_list,ssim_noisy_list,psnr_denoised_list,ssim_denoised_list,steps,step_summary,output_dir,results):
    """
    Saves the metrics in a .csv file for further analysis

    :param psnr_noisy_list:
    :param ssim_noisy_list:
    :param psnr_denoised_list:
    :param ssim_denoised_list:
    :param steps:
    :param step_summary:
    :param output_dir:
    :param results:
    """
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_ssim_noisy = np.mean(ssim_noisy_list)
    avg_psnr_denoised = np.mean(psnr_denoised_list)
    avg_ssim_denoised = np.mean(ssim_denoised_list)
    step_summary.append([steps, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_denoised, avg_ssim_denoised])
    # print the metrics
    print(f"\nStep {steps} Summary:")
    print(f"  Average Noisy   - PSNR: {avg_psnr_noisy:.4f}, SSIM: {avg_ssim_noisy:.4f}")
    print(f"  Average Denoised - PSNR: {avg_psnr_denoised:.4f}, SSIM: {avg_ssim_denoised:.4f}")
    print(f"  Saved to {output_dir}")

    # Save CSV file with the results for possible further analysis
    csv_path = os.path.join(output_dir, f'psnr_ssim_results_step{steps}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'PSNR (Noisy vs Clean)', 'SSIM (Noisy vs Clean)',
            'PSNR (Denoised vs Clean)', 'SSIM (Denoised vs Clean)'
        ])
        writer.writerows(results)


def main(compute_sim_scores):
    """
    Runs the backwards diffusion step using the edm_sampler function from Karras et al. (2022).
    It runs multiple times, increasing the number of diffusion steps by 10 each time.
    It computes the similarity metrics PSNR and SSIM for all runs.

    :param compute_sim_scores:
    :return:
    """
    os.makedirs(output_base_dir, exist_ok=True)
    #loading the network:
    print(f"Loading network from {network_pkl} \n")
    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        network = pickle.load(f)['ema'].to(device)
    network.eval()
    # loading the dataset
    dataset = ImageFolderDataset(input_dir_base, transform=transformer())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Found {len(dataset)} images to denoise. \n")

    step_summary = []
    #denoise using multiple numbers of steps
    for steps in range(10, 101, 10):
        print(f"Starting denoising with {steps} steps")
        output_dir = os.path.join(output_base_dir, f"step_{steps}")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        psnr_denoised_list, ssim_denoised_list = [], []
        psnr_noisy_list, ssim_noisy_list = [], []
        total_images = 0
        total_denoise_time = 0.0

        with torch.no_grad():
            for batch, paths in tqdm(dataloader, desc=f"Steps: {steps}"):
                #load the images as a batch
                latents = batch.to(device)
                #denoise the images and calculate the inference speed
                start_time = time.time()
                denoised = edm_sampler(network, latents, num_steps=steps)
                total_denoise_time += time.time() - start_time
                total_images += len(latents)
                # turn the denoised image back into the correct format and into the correct output subdirectory
                for i in range(denoised.shape[0]):
                    out_img = (denoised[i].detach().cpu().clamp(-1, 1) * 127.5 + 128).permute(1, 2, 0).numpy().astype(np.uint8)
                    relative_path = os.path.relpath(paths[i], input_dir_base)
                    class_name = os.path.dirname(relative_path)
                    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
                    output_path = os.path.join(output_dir, class_name, os.path.basename(paths[i]))
                    Image.fromarray(out_img).save(output_path)

                    #compute similarity scores:
                    if compute_sim_scores:
                        #find the clean images by omitting part of the name not present in the CSV file with the paths.
                        clean_filename = os.path.basename(paths[i]).replace('_x1', '').replace('x1', '')
                        clean_path = os.path.join(clean_dir, class_name, clean_filename)
                        if not os.path.exists(clean_path):
                            continue

                        clean_img = Image.open(clean_path).convert('RGB')
                        clean_tensor = transformer()(clean_img).unsqueeze(0).to(device)
                        degraded_tensor = latents[i].unsqueeze(0).to(device)
                        denoised_tensor = denoised[i].unsqueeze(0).to(device)
                        # compute the PSNR and SSIM of the denoised images vs the clean images and append it to the list
                        psnr_denoised = compute_psnr(clean_tensor, denoised_tensor).item()
                        ssim_denoised = compute_ssim(clean_tensor, denoised_tensor)
                        psnr_denoised_list.append(psnr_denoised)
                        ssim_denoised_list.append(ssim_denoised)
                        # compute the PSNR and SSIM of the degraded images vs the clean images and append it to the list
                        psnr_noisy = compute_psnr(clean_tensor, degraded_tensor).item()
                        ssim_noisy = compute_ssim(clean_tensor, degraded_tensor)
                        psnr_noisy_list.append(psnr_noisy)
                        ssim_noisy_list.append(ssim_noisy)

                        #append the scores per image
                        results.append([os.path.basename(paths[i]), psnr_noisy, ssim_noisy, psnr_denoised, ssim_denoised])

        # save the metrics scores in a csv file.
        # Save the metrics of each number of denoising step
        if compute_sim_scores:
            save_metrics(psnr_noisy_list,ssim_noisy_list,psnr_denoised_list,ssim_denoised_list,steps,step_summary, output_dir, results)
    print("\nAll denoise steps complete.")


# ---------------------- Run ---------------------- #
if __name__ == "__main__":
    # run the denoising algorithm.
    main(compute_sim_scores=True)