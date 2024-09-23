import os
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms 
from skimage.io import imsave
from PIL import Image

from Model_transformer import ImageDeblurringTransformer 
from eval import psnr_between_folders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Define the dataset and dataloader
class DeblurDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        blur_path = self.data.iloc[idx, 1]
        sharp_path = self.data.iloc[idx, 2]
        
        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        
        return blur_img, sharp_img, blur_path, sharp_path
    

data_transform = transforms.Compose([
    transforms.Resize((256, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = DeblurDataset('/home/ninad/vaibhav_r/shubh/test/custom_test/image_test_pairs.csv', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Load the model 
model = ImageDeblurringTransformer().to(device)
checkpoint = torch.load("/home/ninad/vaibhav_r/shubh/checkpoints_transformer/model_epoch_17.pth")
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict)

model.eval()    # Set the model to evaluation mode

# Initialise tqdm progress-bar for evaluation
eval_progress_bar = tqdm(total=len(dataloader), desc="Evaluation", position=0, leave=True)

# Evaluation Loop
generated_image_and_path = []

with torch.no_grad():   # disable gradient-computation during inference
    num_batches = 0

    for i, (blur_imgs, sharp_imgs, blur_path, sharp_path) in enumerate(dataloader):
        blur_imgs = blur_imgs.to(device)        # blur_imgs and sharp_imgs are a "batch" of blur and sharp images respectively,
        sharp_imgs = sharp_imgs.to(device)      # in the form of numpy arrays

        # Forward Pass

        output = model(blur_imgs)
        # Compute the loss

        # Move it to CPU and convert the output tensor to numpy array
        output_numpy = output.cpu().numpy()
        # Create tuples of sharp_path and the corresponding output
        generated_image_and_path.extend( [(path, output) for path, output in zip(sharp_path, output_numpy)] )

        # Update progress bar
        eval_progress_bar.update(1)

# close the progress bar
eval_progress_bar.close()


# create ouput directory if doesn't exist
generated_sharp_folder_path = "/home/ninad/vaibhav_r/shubh/test/custom_test/generated_sharp_images"
if not os.path.exists(generated_sharp_folder_path):
    os.makedirs(generated_sharp_folder_path)
    
# Save the generated sharp images to disk
for path, image in generated_image_and_path:
    image_path = generated_sharp_folder_path + path[-16:]
            # Ensure that the pixel values are in the range [0, 255] before saving
            # You may need to adjust the dtype and scaling of the image accordingly
    image = (image * 255).astype(np.uint8)
    imsave(image_path, image)


sharp_folder_path = "/home/ninad/vaibhav_r/shubh/test/custom_test/sharp"

avg_psnr = psnr_between_folders(sharp_folder_path, generated_sharp_folder_path)
print(f"Average PSNR between corresponding images: {avg_psnr} dB")