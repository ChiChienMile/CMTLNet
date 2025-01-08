import torch

# Check if CUDA (GPU) is available and set the device accordingly
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Number of worker threads for data loading
n_threads = 8

# Number of processes for multiprocessing (if applicable)
num_processes = 5

# Base batch size for training each individual task.
# For each task, we apply oversampling by selecting one sample with label 0 and one sample with label 1 per batch.
# The effective total batch size is batch_size * 6, accounting for the tasks 1p19q, IDH, and LHG.
batch_size = 1

# Random seed for reproducibility
random_seed = 1337

# Learning rate for the optimizer
lr_self = 1e-4

# Minimum learning rate for the learning rate scheduler
min_lr = 1e-6

# Weight decay (L2 regularization) to prevent overfitting
weight_decay = 2e-5

# Factor by which the learning rate will be reduced
lr_reduce_factor = 0.7

# Number of epochs with no improvement after which the learning rate will be reduced
lr_reduce_patience = 8

# Number of epochs for the first training stage
epochs_s1 = 60

# Number of epochs for the second training stage
epochs_s2 = 40

# Epoch number from which to start saving the model
save_epoch = 0

# Directory path where data and results will be stored
data_dir = './'

# Flag to enable or disable random operations (e.g., data shuffling)
random_flag = True
