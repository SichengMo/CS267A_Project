import os

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
from itertools import product
torch.autograd.set_detect_anomaly(True)
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce



# Set the path to the MovieLens-100K folder
folder_path = 'ml-100k'

# Load the MovieLens-100K dataset
train_files = [os.path.join(folder_path, f'u{i}_combined.base') for i in range(1, 5)]
test_files = [os.path.join(folder_path, f'u{i}_combined.test') for i in range(1, 5)]
final_test_file = os.path.join(folder_path, 'u5_combined.test')

# Set the hyperparameters to be searched
hidden_dim_values = [128]
lr_values = [0.01]
num_epochs_values = [50]
batch_size_values = [32]

# Perform 4-fold cross-validation and grid search
best_rmse = float('inf')
best_params = {}
from torch.nn.functional import normalize
names = ['user_id', 'item_id', 'rating', 'timestamp']
combined_names = ['user_id', 'item_id', 'rating', 'timestamp', 'gener1', 'gener2', 'gener3', 'director', 'cast1',
                  'cast2', 'cast3', 'cast4', 'runtime', 'language', 'country', 'company']

test_name = ['user_id', 'item_id','gener1','gener2','gener3']
#test_name = ['user_id', 'item_id','gener1','gener2','gener3','director']
#test_name = ['user_id', 'item_id','gener1','gener2','gener3','director','cast1','cast2','cast3','cast4']
#test_name = ['user_id', 'item_id','gener1','gener2','gener3','director','cast1','cast2','cast3','cast4','country']
#test_name = ['user_id', 'item_id','gener1','gener2','gener3','director','cast1','cast2','cast3','cast4','country','runtime','language', 'company']
# best_params['hidden_dim'] = 128
# best_params['lr'] = 0.01
# best_params['num_epochs'] = 200
# best_params['batch_size'] = 256

class SoftmaxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SoftmaxModel, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,10)
        self.fc3 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

enc = OneHotEncoder(handle_unknown='ignore')
#enc = ce.BinaryEncoder(cols=test_name, return_df=False)
# for hidden_dim, lr, num_epochs, batch_size in product(hidden_dim_values, lr_values, num_epochs_values,
#                                                       batch_size_values):
#     print(f"Training model with hidden_dim={hidden_dim}, lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}")
#
#     rmses = []
#     num_folds = 4
#     fold = 0
#     for train_file, test_file in zip(train_files, test_files):
#         if 'combined' in train_file:
#             train_data = pd.read_csv(train_file, delimiter='\t', names=combined_names)
#             test_data = pd.read_csv(test_file, delimiter='\t', names=combined_names)
#         else:
#             # Load the train and test data for this fold
#             train_data = pd.read_csv(train_file, delimiter='\t', names=names)
#             test_data = pd.read_csv(test_file, delimiter='\t', names=names)
#
#         # Stack the training and test data
#         combined_data = pd.concat([train_data[test_name], test_data[test_name]])
#
#         #enc = ce.BinaryEncoder(cols=combined_names, return_df=True)
#         # One-hot encode the combined data
#         combined_data_enc = enc.fit_transform(combined_data)
#
#         # Convert to PyTorch tensors
#         combined_features = torch.tensor(combined_data_enc.toarray(), dtype=torch.float32)
#
#         input_dim = combined_features.shape[-1]
#         # Split the features and labels back into training and test sets
#         train_size = len(train_data)
#         train_features = combined_features[:train_size]
#         test_features = combined_features[train_size:]
#
#         train_labels = torch.tensor(train_data['rating'].values - 1,
#                                     dtype=torch.long)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]
#         test_labels = torch.tensor(test_data['rating'].values - 1,
#                                    dtype=torch.long)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]
#         output_dim = 5
#         # Create DataLoader for training data
#         train_loader = DataLoader(
#             TensorDataset(train_features, train_labels),
#             batch_size=batch_size,
#             shuffle=True
#         )
#
#         # Create DataLoader for test data
#         test_loader = DataLoader(
#             TensorDataset(test_features, test_labels),
#             batch_size=batch_size,
#             shuffle=False
#         )
#
#         # Initialize the model
#         model = SoftmaxModel(input_dim, hidden_dim, output_dim)
#
#         # Define the loss function
#         criterion = nn.CrossEntropyLoss()
#
#         # Define the optimizer
#         optimizer = optim.SGD(model.parameters(), lr=lr)
#
#         # Training loop
#         for epoch in range(num_epochs):
#             model.train()
#             total_loss = 0.0
#             for inputs, targets in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 # print(targets)
#                 # print(outputs)
#                 # outputs = torch.clamp(outputs, min=1e-4, max=1 - 1e-4)
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 #torch.nn.utils.clip_grad_norm_(model.parameters(),  1)
#                 optimizer.step()
#                 total_loss += loss.item()
#             print(
#                 f'Fold {fold + 1}/{num_folds}, Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')
#
#         # Evaluation on the test set
#         model.eval()
#         predictions = []
#         true_labels = []
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, dim=1)
#                 predictions.extend(predicted.tolist())
#                 true_labels.extend(targets.tolist())
#
#         # Calculate RMSE for this fold
#         rmse = mean_squared_error(true_labels, predictions, squared=False)
#         rmses.append(rmse)
#         fold += 1
#     # Calculate the mean RMSE across folds
#     mean_rmse = sum(rmses) / len(rmses)
#
#     # Check if this set of hyperparameters is the best so far
#     if mean_rmse < best_rmse:
#         best_rmse = mean_rmse
#         best_params['hidden_dim'] = hidden_dim
#         best_params['lr'] = lr
#         best_params['num_epochs'] = num_epochs
#         best_params['batch_size'] = batch_size
#
#     print('Best Parameters: ', best_params)


# Evaluation on the final test set using the best parameters
final_test_data = pd.read_csv(final_test_file, delimiter='\t', names=combined_names)

# Stack the final test data with the entire training data (all folds)
# all_train_data = pd.concat([pd.read_csv(f, delimiter='\t', names=combined_names) for f in train_files])
all_train_data = pd.read_csv(os.path.join(folder_path, 'u5_combined.base'), delimiter='\t', names=combined_names)
combined_data = pd.concat([all_train_data[test_name], final_test_data[test_name]])

# One-hot encode the combined data
combined_data_enc = enc.fit_transform(combined_data)

# Convert to PyTorch tensors
if type(combined_data_enc) == numpy.ndarray:
    combined_features = torch.from_numpy(combined_data_enc).to(torch.float32)
else:
    combined_features = torch.tensor(combined_data_enc.toarray(), dtype=torch.float32)
# print(combined_features.shape)
input_dim = combined_features.shape[-1]
print(input_dim)
output_dim = 5
# Split the features and labels back into training and final test sets
train_size = len(all_train_data)
train_features = combined_features[:train_size]
# train_labels = combined_labels[:train_size]
test_features = combined_features[train_size:]
# test_labels = combined_labels[train_size:]

train_labels = torch.tensor(all_train_data['rating'].values - 1,
                            dtype=torch.long)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]
test_labels = torch.tensor(final_test_data ['rating'].values - 1,
                           dtype=torch.long)  # Subtract 1 to adjust score to [0, 1, 2, 3, 4]

# Initialize the model with the best parameters
model = SoftmaxModel(input_dim, best_params['hidden_dim'], output_dim)

# Define the optimizer with the best learning rate
optimizer = optim.SGD(model.parameters(), lr=best_params['lr'])
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=best_params['batch_size'],
            shuffle=True
        )

# Create DataLoader for test data
test_loader = DataLoader(
    TensorDataset(test_features, test_labels),
    batch_size=2000,
    drop_last=False,
    shuffle=False
)


# Training loop
model.cuda()
for epoch in range(best_params['num_epochs']):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.cuda()).cpu()
        loss = criterion(outputs.squeeze(0), targets.squeeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}/{best_params["num_epochs"]}, Loss: {total_loss / len(train_loader)}')

# Evaluation on the final test set
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs.cuda()).cpu()
        _, predicted = torch.max(outputs, dim=1)
        predictions.extend(predicted.tolist())
        true_labels.extend(targets.tolist())

# Calculate the final metrics
final_rmse = mean_squared_error(true_labels, predictions, squared=False)
accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions, average='macro')
precision = precision_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')

print(f'Final RMSE: {final_rmse}')
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
