import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread

def psnr_between_folders(sharp_folder_path: str, blur_folder_path: str) -> float:
    psnr_values = []
    
    for file_name in os.listdir(sharp_folder_path):
        if file_name.endswith(('jpg','jpeg','png')):

            # Read corresponding images from both folders
            sharp_image_path = os.path.join(sharp_folder_path, file_name)
            blur_image_path = os.path.join(blur_folder_path, file_name)
            img1 = imread(sharp_image_path)
            img2 = imread(blur_image_path)
            
            # Compute PSNR between corresponding images
            psnr = peak_signal_noise_ratio(img1, img2)
            psnr_values.append(psnr)
    
    # Compute average PSNR across all images
    avg_psnr = sum(psnr_values) / len(psnr_values)
    # print (len(psnr_values))
    
    return avg_psnr

# # Example usage:
# sharp_folder_path = "/home/shubhranil/shubh/miniproject2_gnr/data/test/sharp"
# blur_folder_path = "/home/shubhranil/shubh/miniproject2_gnr/data/test/blur"

# avg_psnr = psnr_between_folders(sharp_folder_path, blur_folder_path)
# print(f"Average PSNR between corresponding images: {avg_psnr} dB")
