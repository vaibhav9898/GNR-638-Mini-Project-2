import torch, random
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale
from torch.nn.functional import mse_loss
import torchvision.models as models
from PIL import Image
import pandas as pd
from Model_transformer import ImageDeblurringTransformer
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ImageDeblurringDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        sharp_path = self.data.iloc[idx, 2]
        image = Image.open(img_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            sharp_img = self.transform(sharp_img)
        return image,sharp_img

# Define transformations for data augmentation or normalization
data_transform = transforms.Compose([
    transforms.Resize((256, 448)),  # Resize images to desired size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# Normalize images
])

# Define paths
csv_file_path = '/home/ninad/vaibhav_r/shubh/image_pairs.csv'
test_csv="/home/ninad/vaibhav_r/shubh/image_pairs_next1k.csv"
# Create dataset instance
dataset = ImageDeblurringDataset(csv_file=csv_file_path, transform=data_transform)
test_dataset=ImageDeblurringDataset(csv_file=test_csv, transform=data_transform)
# Create DataLoader instance for batching and shuffling
batch_size = 2
shuffle = True
num_workers = 4  # Number of subprocesses to use for data loading
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
# test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
# Initialize the model
model = ImageDeblurringTransformer().to(device)

###########
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()

        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        return h_relu1

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.l1 = nn.L1Loss()
        self.ab = ablation
        self.down_sample_4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear')
    def forward(self, restore, sharp, blur):
        B, C, H, W = restore.size()
        restore_vgg, sharp_vgg, blur_vgg = self.vgg(restore), self.vgg(sharp), self.vgg(blur)

        # filter out sharp regions
        threshold = 0.01
        mask = torch.mean(torch.abs(sharp-blur), dim=1).view(B, 1, H, W)
        mask[mask <= threshold] = 0
        mask[mask > threshold] = 1
        mask = self.down_sample_4(mask)
        d_ap = torch.mean(torch.abs((restore_vgg - sharp_vgg.detach())), dim=1).view(B, 1, H//4, W//4)
        d_an = torch.mean(torch.abs((restore_vgg - blur_vgg.detach())), dim=1).view(B, 1, H//4, W//4)
        mask_size = torch.sum(mask)
        contrastive = torch.sum((d_ap / (d_an + 1e-7)) * mask) / mask_size

        return contrastive


class ContrastLoss_Ori(nn.Module):
    def __init__(self, ablation=False):
        super(ContrastLoss_Ori, self).__init__()
        self.vgg = Vgg19().to(device)
        self.l1 = nn.L1Loss()
        self.ab = ablation

    def forward(self, restore, sharp, blur):

        restore_vgg, sharp_vgg, blur_vgg = self.vgg(restore), self.vgg(sharp), self.vgg(blur)
        d_ap = self.l1(restore_vgg, sharp_vgg.detach())
        d_an = self.l1(restore_vgg, blur_vgg.detach())
        contrastive_loss = d_ap / (d_an + 1e-7)
        
        return contrastive_loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to(device)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        # x = torch.clamp(x + 0.5, min = 0,max = 1)
        # y = torch.clamp(y + 0.5, min = 0,max = 1)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class Stripformer_Loss(nn.Module):

    def __init__(self, ):
        super(Stripformer_Loss, self).__init__()

        self.char = CharbonnierLoss()
        self.edge = EdgeLoss()
        self.contrastive = ContrastLoss()

    def forward(self, restore, sharp, blur):
        char = self.char(restore, sharp)
        edge = 0.05 * self.edge(restore, sharp)
        contrastive = 0.0005 * self.contrastive(restore, sharp, blur)
        loss = char + contrastive + edge 
        return loss

############

loss_fn=Stripformer_Loss()
#loss_fn = nn.L1Loss(reduction="mean")

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999))

####
checkpoint_path = "/home/ninad/vaibhav_r/shubh/checkpoints_transformer/model_epoch_striploss_8.pth"
checkpoint = torch.load(checkpoint_path)
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']

model.load_state_dict(model_state_dict)
model.to(device)
optimizer.load_state_dict(optimizer_state_dict)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

####
checkpoint_dir = 'checkpoints_transformer'
os.makedirs(checkpoint_dir, exist_ok=True)
#test_blur,test_sharp=test_dataloader
# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    for batch_idx, (images,sharp_imgs) in enumerate(data_loader):
        images = images.to(device)
        sharp_imgs = sharp_imgs.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate perceptual loss
        loss=loss_fn(outputs,sharp_imgs,images)
        # loss = loss_fn(outputs, sharp_imgs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Print statistics
        if batch_idx % 10 == 0:  # Print every 10 optimizer step             
            image = (outputs[0] * 255).cpu().detach().numpy().astype(np.uint8)  # Convert to NumPy array
            sharp_img_np = (sharp_imgs[0] * 255).cpu().detach().numpy().astype(np.uint8)  # Convert to NumPy array
            psnr = peak_signal_noise_ratio(image, sharp_img_np)  # Calculate PSNR
            print(f'Epoch: {epoch}/{num_epochs}, Step: {batch_idx}/{data_loader.__len__()}, Train Loss: {loss.item():.3f}, Train PSNR: {psnr:.3f}')
            
            random_number = random.choice(range(test_dataset.__len__()))
            test_blur, test_sharp = test_dataset[random_number]

            test_blur=test_blur.to(device)
            test_sharp=test_sharp.to(device)
            restore=(model(test_blur.unsqueeze(0)).squeeze(0)*255).cpu().detach().numpy().astype(np.uint8)
            sharp_img_np=(test_sharp*255).cpu().detach().numpy().astype(np.uint8)
            psnr = peak_signal_noise_ratio(restore, sharp_img_np)
            print(f"Epoch: {epoch}/{num_epochs}, Test PSNR: {psnr:.3f}")
        
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_striploss_full{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

print('Finished Training')