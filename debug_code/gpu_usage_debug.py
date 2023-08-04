import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'



class YourModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(YourModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data_length=10000):
        self.data_length = data_length

        self.count = 0
        
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index):
        lr = torch.rand(1,200,200)
        hr = torch.rand(1,200,200)
        self.count += 1
        # print("taking data", self.count)
        return {'input': lr, 'label': hr}


dataset = CustomDataset()

# Create DataLoader for batch processing
batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define your model
model = YourModel()

'''wrap model for data parallelism'''
num_of_gpus = torch.cuda.device_count()
print("Number of GPU available", num_of_gpus)
if num_of_gpus>1:
    generator = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
    print("Multiple GPU Training")


# Define your loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10000

for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_samples = 0
    
    for batch in dataloader:
        inputs = batch['input']
        labels = batch['label']
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update epoch loss and samples count
        epoch_loss += loss.item() * len(inputs)
        epoch_samples += len(inputs)

        print("Finished one iteration")
        print("***************************************************************************8")
        
    # Calculate average epoch loss
    average_loss = epoch_loss / epoch_samples
    
    # Print epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}")
    
    # Track memory consumption
    memory_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else torch.cuda.memory_allocated()
    memory_cached = torch.cuda.memory_cached() if torch.cuda.is_available() else torch.cuda.memory_cached()
    print(f"Memory Allocated: {memory_allocated} bytes, Memory Cached: {memory_cached} bytes")