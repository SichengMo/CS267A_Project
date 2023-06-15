import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
from itertools import product
torch.autograd.set_detect_anomaly(True)

# Set the path to the MovieLens-100K folder
folder_path = 'ml-100k'

# Load the MovieLens-100K dataset
train_files = [os.path.join(folder_path, f'u{i}_combined.base') for i in range(1, 5)]
test_files = [os.path.join(folder_path, f'u{i}_combined.test') for i in range(1, 5)]
final_test_file = os.path.join(folder_path, 'u5_combined.test')

# Set the hyperparameters to be searched
hidden_dim_values = [600]
lr_values = [0.001]
num_epochs_values = [20]
batch_size_values = [256]

# Perform 4-fold cross-validation and grid search
best_rmse = float('inf')
best_params = {}
from torch.nn.functional import normalize
names = ['user_id', 'item_id', 'rating', 'timestamp']
combined_names = ['user_id', 'item_id', 'rating', 'timestamp', 'gener1', 'gener2', 'gener3', 'director', 'cast1',
                  'cast2', 'cast3', 'cast4', 'runtime', 'language', 'country', 'company']


test_name = ['user_id', 'item_id','timestamp']

class SoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SoftmaxModel, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print(x[0])
        #x = self.bn1(x)
        #x = self.bn1(x)
        # print(x[0])
        x = torch.relu(self.fc1(x))
        # print(x[0])
        x = self.fc2(x)
        # print(x[0])
        # x = nn.Sigmoid()(x)
        return x


for hidden_dim, lr, num_epochs, batch_size in product(hidden_dim_values, lr_values, num_epochs_values,
                                                      batch_size_values):
    print(f"Training model with hidden_dim={hidden_dim}, lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}")
    rmses = []
    num_folds = 4
    fold = 0
    for train_file, test_file in zip(train_files, test_files):
        if 'combined' in train_file:
            train_data = pd.read_csv(train_file, delimiter='\t', names=combined_names)
            test_data = pd.read_csv(test_file, delimiter='\t', names=combined_names)
        else:
            # Load the train and test data for this fold
            train_data = pd.read_csv(train_file, delimiter='\t', names=names)
            test_data = pd.read_csv(test_file, delimiter='\t', names=names)

        # Split the data into features and labels
        train_features = torch.tensor(train_data[test_name].values, dtype=torch.float32)

        train_labels = torch.tensor(train_data['rating'].values - 1,
                                    dtype=torch.float32)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]
        test_features = torch.tensor(test_data[test_name].values, dtype=torch.float32)
        test_labels = torch.tensor(test_data['rating'].values - 1,
                                   dtype=torch.float32)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]
        train_features = normalize(train_features,dim=-1)
        test_features = normalize(test_features,dim=-1)

        input_dim = len(test_name)
        output_dim = 1
        # Create DataLoader for training data
        train_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=batch_size,
            shuffle=True
        )

        # Create DataLoader for test data
        test_loader = DataLoader(
            TensorDataset(test_features, test_labels),
            batch_size=batch_size,
            shuffle=False
        )

        # Initialize the model
        model = SoftmaxModel(input_dim, hidden_dim, output_dim)

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                # print(targets)
                # print(outputs)
                # outputs = torch.clamp(outputs, min=1e-4, max=1 - 1e-4)
                loss = criterion(outputs.squeeze(0), targets.squeeze(0))
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(),  1)
                optimizer.step()
                total_loss += loss.item()
            print(
                f'Fold {fold + 1}/{num_folds}, Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        # Evaluation on the test set
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                predictions.extend(predicted.tolist())
                true_labels.extend(targets.tolist())

        # Calculate RMSE for this fold
        rmse = mean_squared_error(true_labels, predictions, squared=False)
        rmses.append(rmse)
        fold += 1
    # Calculate the mean RMSE across folds
    mean_rmse = sum(rmses) / len(rmses)

    # Check if this set of hyperparameters is the best so far
    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_params['hidden_dim'] = hidden_dim
        best_params['lr'] = lr
        best_params['num_epochs'] = num_epochs
        best_params['batch_size'] = batch_size

    print('Best Parameters: ', best_params)
# Use the best hyperparameters to train the final model on 'b5'
train_data = pd.read_csv(os.path.join(folder_path, 'u5_combined.base'), delimiter='\t',
                         names=combined_names)
test_data = pd.read_csv(final_test_file, delimiter='\t', names=combined_names)

train_features = torch.tensor(train_data[test_name].values, dtype=torch.float32)
train_labels = torch.tensor(train_data['rating'].values - 1,
                            dtype=torch.float32)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]
test_features = torch.tensor(test_data[test_name].values, dtype=torch.float32)
test_labels = torch.tensor(test_data['rating'].values - 1,
                           dtype=torch.float32)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]

train_loader = DataLoader(
    TensorDataset(train_features, train_labels),
    batch_size=best_params['batch_size'],
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(test_features, test_labels),
    batch_size=best_params['batch_size'],
    shuffle=False
)

# Initialize the final model with the best hyperparameters
model = SoftmaxModel(input_dim, best_params['hidden_dim'], output_dim)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

# Train the final model on 'b5'
for epoch in range(best_params['num_epochs']):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    num_epochs = best_params['num_epochs']
    print(f'Final Model - Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

# Evaluate the final model on the test set 'b5'
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        predictions.extend(predicted.tolist())
        true_labels.extend(targets.tolist())

# Calculate evaluation metrics for the final model
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
precision = precision_score(true_labels, predictions, average='weighted')
rmse = mean_squared_error(true_labels, predictions, squared=False)

# Print the evaluation metrics for the final model
print(best_params)
print("Final Model Evaluation (b5)")
print(f'Test Accuracy: {accuracy}')
print(f'Test F1 Score: {f1}')
print(f'Test Recall: {recall}')
print(f'Test Precision: {precision}')
print(f'Test RMSE: {rmse}')
