import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utils import *
from model import UNET

def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data): 
        X, y = batch[0], batch[1]
        X, y = X.to(device), y.to(device)
        preds = model(X)
    
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def main():

    if torch.cuda.is_available():
        device = 'cuda:0'
        print("Running on GPU")
    else:
        device = 'cpu'
        print("Running on CPU")

    model_path = 'model'
    root_dir = 'dataset/gtFine'
    img_h = 110
    img_w = 220
    batch_size = 4
    lr = 1e-3
    epochs = 5
    unet = UNET(in_channels=3, out_channels=19-1).to(device).train()
    # print(unet)
    transform = transforms.Compose([transforms.Resize((img_h, img_w), interpolation=Image.NEAREST)])
    train_set = get_cityscapes_data(split='train',mode='fine',relabelled=True,root_dir=root_dir,transforms=transform,batch_size=batch_size)
    print('Data loaded succesfully!')
    unet = UNET(in_channels=3, out_channels=19).to(device).train()
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss(ignore_index=255) 
    LOSS_VALS = list()
    for e in range(epochs):
        print(f'Epoch: {e}')
        loss_val = train_function(train_set, unet, optimizer, loss_function, device)
        LOSS_VALS.append(loss_val) 
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': LOSS_VALS
        }, model_path)
        print("Epoch completed and model successfully saved!")
    

if __name__ == '__main__':
    main()


