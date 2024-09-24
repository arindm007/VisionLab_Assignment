import re
import matplotlib.pyplot as plt

# Load the content of the log file
with open('log.txt', 'r') as file:
    log_data = file.readlines()

# Initialize lists to hold extracted train and test losses
train_losses = []
test_losses = []
epochs = []

# Regex to find train and test loss values
train_loss_pattern = re.compile(r'"train_loss": ([\d.]+)')
test_loss_pattern = re.compile(r'"test_loss": ([\d.]+)')
epoch_pattern = re.compile(r'"epoch": ([\d.]+)')

# Loop through the log data and extract losses
for line in log_data:
    train_match = train_loss_pattern.search(line)
    test_match = test_loss_pattern.search(line)
    epoch_match = epoch_pattern.search(line)

    if train_match:
        train_losses.append(float(train_match.group(1)))
    if test_match:
        test_losses.append(float(test_match.group(1)))
    if epoch_match:
        epochs.append(float(epoch_match.group(1)))





# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

