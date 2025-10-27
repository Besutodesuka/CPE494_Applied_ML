import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import os
import logging
from ann import Net
logger = logging.getLogger()
# 1. define scaling function
def scale(data, from_interval: Tuple[float, float], to_interval: Tuple[float, float]=(0, 1)):
    from_min, from_max = from_interval
    to_min, to_max = to_interval
    scaled_data = to_min + (data - from_min) * (to_max - to_min) / (from_max - from_min)
    return scaled_data

# 2. read data
dataframe = pd.read_csv('history_all.csv', sep=',')
dataframe.iloc[:, 0:8] = scale(dataframe.iloc[:, 0:8], from_interval=(0, 100), to_interval=(0,1)) # distance
dataframe.iloc[:, 8] = scale(dataframe.iloc[:, 8], from_interval=(-180, 180), to_interval=(0,1)) # scale smell feature
dataframe.iloc[:, 9] = scale(dataframe.iloc[:, 9], from_interval=(-10, 10), to_interval=(0,1))
dataframe.iloc[:, 10] = scale(dataframe.iloc[:, 10], from_interval=(-180, 180), to_interval=(0,1))

# 3. select input and output of the ANN
x = dataframe.iloc[:, :9].values
y = dataframe.iloc[:, 9:].values

print(x, y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
# 4. define ANN architecture, loss and optimizer
model = Net()
if os.path.isfile('assignment5_model.pth'):
    model.load_state_dict(torch.load('assignment5_model.pth'))
    logger.info("Loaded pre-trained model.")



criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. train ANN model
epochs = 1000
batch_size = 1024
losses = []

from tqdm import tqdm

for epoch in tqdm(range(epochs)):
    for i in range(0, len(X_tensor), batch_size):
        # Get batch
        X_batch = X_tensor[i:i+batch_size]
        y_batch = y_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# 6. save model for later usage
torch.save(model.state_dict(), 'assignment5_model.pth')
logger.info("Saved trained model to assignment5_model.pth")

# 7. plot the loss value of the training
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
