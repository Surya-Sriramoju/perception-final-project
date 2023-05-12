import torch
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader

# Define the transform to apply to the data
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Initialize the Cityscapes dataset with the desired split and transform
cityscapes_train = Cityscapes(root='dataset/gtFine', split='train', mode='fine', target_type='semantic', transform=transform)

# Use the DataLoader to create batches of data
train_loader = DataLoader(cityscapes_train, batch_size=32, shuffle=True)

# Define your model and loss function
# model = MyModel()
criterion = torch.nn.CrossEntropyLoss()

# Define your optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels from the DataLoader
        inputs, labels = data
        print(inputs.shape)

        # Zero the parameter gradients
    

print('Finished Training')