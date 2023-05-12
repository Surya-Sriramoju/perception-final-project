import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import Cityscapes
from torchsummary import summary


# Define the Vision Transformer-based segmentation model
class Vision_Transformer(nn.Module):
    def __init__(self, number_of_classes, patch_size=16, hidden_dim=768, number_of_layers=12, number_of_heads=12):
        super(Vision_Transformer, self).__init__()
        
        # Pre-trained Vision Transformer as the backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc  # Remove the fully connected layer
        
    
        self.positional_encoding = nn.Embedding(patch_size * patch_size, hidden_dim)
        
        #segmentation head is not correct right now
        # self.segmentation_head = nn.Sequential(
        #     nn.Conv2d(2048, hidden_dim, kernel_size=1),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #     nn.Conv2d(hidden_dim, number_of_classes, kernel_size=1)
        
        
       
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=number_of_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=number_of_layers)
        
    def feedforward(self, x):
      
        features = self.backbone(x)
        
      
        patches = features.flatten(2).permute(2, 0, 1)
        patch_ids = torch.arange(patches.size(0)).unsqueeze(1).to(patches.device)
        patch_embeddings = self.positional_encoding(patch_ids) + patches
        
     
        encoded_patches = self.transformer(patch_embeddings)
        
      
        encoded_patches = encoded_patches.permute(1, 2, 0).view(-1, features.size(2), features.size(3), features.size(1))
        segmentation_output = self.segmentation_head(encoded_patches)
        
        return segmentation_output


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(dataloader.dataset)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Vision_Transformer(number_of_classes=19).to(device)

train_dataset = Cityscapes(root='path/to/cityscapes', split='train', mode='fine', target_type='semantic',
                           transform=ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
