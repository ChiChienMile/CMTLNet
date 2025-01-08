import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import config
import monai
import warnings
import numpy as np
import pandas as pd
from model.CMTLNet import CFE
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataLoder.loader_S1 import TrainDataLoader_balance as TrainDataLoader
from dataLoder.loader_S1 import TestDataLoader
from utiles import save_model, get_learning_rate, calculate_classification_metrics, move_tensor_to_device, AverageMeter

def validate(data_loader, model, mode_flag, ssl_index, class_name):
    """
    Validates the model on a given dataset and computes loss and classification metrics.

    Parameters:
        data_loader (DataLoader): DataLoader for the dataset to validate.
        model (torch.nn.Module): The model to validate.
        mode_flag (str): Indicates whether the dataset is 'Train' or 'Test'.
        ssl_index (int): SSL index used for model prediction.
        class_name (str): Name of the class/category being validated.

    Returns:
        tuple: Contains average loss, accuracy, F1 score, precision, and recall.
    """
    model.eval()  # Set model to evaluation mode
    cross_entropy_loss = CrossEntropyLoss(reduction='mean')  # Define cross-entropy loss
    all_targets, all_predictions = [], []  # Lists to store all true labels and predictions
    loss_meter = AverageMeter()  # Utility to track average loss

    for batch_data in data_loader:
        images, labels = batch_data
        images = move_tensor_to_device(images, use_gpu=config.use_cuda)  # Move images to GPU if available
        labels = move_tensor_to_device(labels, use_gpu=config.use_cuda)  # Move labels to GPU if available

        with torch.no_grad():  # Disable gradient computation
            outputs = model.predict(images, ssl_index)  # Get model predictions
            loss = cross_entropy_loss(outputs, labels)  # Compute loss
            loss_meter.update(loss.item(), 1)  # Update loss meter
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Compute probabilities
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()  # Get predicted classes
            true_labels = labels.cpu().detach().numpy()  # Get true labels
            all_predictions.extend(predicted_classes)  # Collect predictions
            all_targets.extend(true_labels)  # Collect true labels

    average_loss = loss_meter.average  # Calculate average loss
    all_predictions = np.array(all_predictions, dtype=np.int32)
    all_targets = np.array(all_targets, dtype=np.int32)
    # Compute classification metrics: Accuracy, F1, Precision, Recall, and Confusion Matrix
    accuracy, f1, precision, recall, confusion_matrix = calculate_classification_metrics(
        predictions=all_predictions,
        targets=all_targets,
        average_method="macro"
    )
    print(confusion_matrix)  # Print confusion matrix

    # Print metrics based on whether it's training or testing
    if mode_flag == 'Test':
        print(f'Test set {class_name} | Accuracy {accuracy:.4f} | F1 {f1:.4f} | Precision {precision:.4f} | Recall {recall:.4f} |')
    else:
        print(f'Train set {class_name} | Accuracy {accuracy:.4f} | F1 {f1:.4f} | Precision {precision:.4f} | Recall {recall:.4f} |')

    model.train()  # Set model back to training mode

    return average_loss, accuracy, f1, precision, recall


def valid_metrics_by_class(model, train_loader, test_loader, ssl_index, class_name):
    """
    Computes validation metrics for a specific class on both training and testing datasets.

    Parameters:
        model (torch.nn.Module): The model to validate.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        ssl_index (int): SSL index used for model prediction.
        class_name (str): Name of the class/category being validated.

    Returns:
        tuple: Contains training metrics and testing metrics.
    """
    # Validate on training data
    train_loss, train_accuracy, train_f1, train_precision, train_recall = validate(
        data_loader=train_loader,
        model=model,
        mode_flag='Train',
        ssl_index=ssl_index,
        class_name=class_name
    )

    # Validate on testing data
    test_loss, test_accuracy, test_f1, test_precision, test_recall = validate(
        data_loader=test_loader,
        model=model,
        mode_flag='Test',
        ssl_index=ssl_index,
        class_name=class_name
    )

    # Collect training and testing metrics
    metrics_train = [train_loss, train_accuracy, train_f1, train_precision, train_recall]
    metrics_test = [test_loss, test_accuracy, test_f1, test_precision, test_recall]
    return metrics_train, metrics_test


def get_pandas_series(epoch, train_metrics, test_metrics):
    """
    Creates a Pandas Series for logging metrics.

    Parameters:
        epoch (int): Current epoch number.
        train_metrics (list): List of training metrics.
        test_metrics (list): List of testing metrics.

    Returns:
        pd.Series: Series containing epoch and corresponding metrics.
    """
    series = pd.Series([
        epoch,
        train_metrics[0],  # Training loss
        train_metrics[1],  # Training accuracy
        train_metrics[2],  # Training F1 score
        test_metrics[0],   # Testing loss
        test_metrics[1],   # Testing accuracy
        test_metrics[2],   # Testing F1 score
    ], index=[
        'epoch',
        'Train_loss', 'Train_Accuracy', 'Train_F1',
        'Val_loss', 'Val_Accuracy', 'Val_F1'
    ])
    return series


def get_class_loaders(TrainDataset, TestDataset, transforms, fold, loader_id, pkl_path):
    """
    Creates DataLoaders for training and validation datasets for a specific class.

    Parameters:
        TrainDataset (Dataset): Training dataset class.
        TestDataset (Dataset): Testing dataset class.
        transforms (Compose): Data transformation pipeline.
        fold (int): Current fold number for cross-validation.
        loader_id (str): Identifier for the class.
        pkl_path (str): Path to the pickle file containing dataset information.

    Returns:
        tuple: Contains training DataLoader, validation training DataLoader, and validation testing DataLoader.
    """
    # Create training dataset
    train_dataset = TrainDataset(
        mode='train',
        fold=fold,
        transform=transforms,
        random_flag=config.random_flag,
        pickle_path=pkl_path
    )

    # Create training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print(f"Number of train_loader {loader_id} examples {len(train_dataset)}")

    # Create validation training dataset
    val_train_dataset = TestDataset(
        mode='train',
        fold=fold,
        transform=transforms,
        pickle_path=pkl_path
    )

    # Create validation training DataLoader
    val_train_loader = DataLoader(
        val_train_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print(f"Number of val_train_loader {loader_id} examples {len(val_train_dataset)}")

    # Create validation testing dataset
    val_test_dataset = TestDataset(
        mode='test',
        fold=fold,
        transform=transforms,
        pickle_path=pkl_path
    )

    # Create validation testing DataLoader
    val_loader = DataLoader(
        val_test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_threads,
        pin_memory=config.use_cuda
    )
    print(f"Number of val_loader {loader_id} examples {len(val_test_dataset)}")
    return train_loader, val_train_loader, val_loader

def reshape_input_data(data):
    """
    Reshapes input data and moves it to the appropriate device (GPU or CPU).

    Parameters:
        data (tuple): Tuple containing images, tumour-related labels, and classification labels.

    Returns:
        tuple: Reshaped images, tumour-related labels, and classification labels.
    """
    images, label_NT, labels = data
    # Reshape images
    images = images.view(
        images.size(0) * images.size(1),
        images.size(2),
        images.size(3),
        images.size(4),
        images.size(5)
    )
    # Reshape non-target labels and target labels
    label_NT = label_NT.view(-1)
    labels = labels.view(-1)

    # Move data to device
    images = move_tensor_to_device(images, use_gpu=config.use_cuda)
    label_NT = move_tensor_to_device(label_NT, use_gpu=config.use_cuda)
    labels = move_tensor_to_device(labels, use_gpu=config.use_cuda)
    return images, label_NT, labels


def train(fold, pkl_paths):
    """
    Main training function responsible for the entire training process, including data loading, model training,
    validation, logging, and model saving.

    Parameters:
        fold (int): Current fold number for cross-validation.
        pkl_paths (list): List of paths to pickle files for each class.
    """
    # Define data transformations
    val_transforms = monai.transforms.Compose([
        monai.transforms.ToTensor(),
    ])

    # Get DataLoaders for each class
    train_loader_1p19q, val_train_loader_1p19q, val_loader_1p19q = get_class_loaders(
        TrainDataset=TrainDataLoader,
        TestDataset=TestDataLoader,
        transforms=val_transforms,
        fold=fold,
        loader_id='1p19q',
        pkl_path=pkl_paths[0]
    )
    train_loader_IDH, val_train_loader_IDH, val_loader_IDH = get_class_loaders(
        TrainDataset=TrainDataLoader,
        TestDataset=TestDataLoader,
        transforms=val_transforms,
        fold=fold,
        loader_id='IDH',
        pkl_path=pkl_paths[1]
    )
    train_loader_LHG, val_train_loader_LHG, val_loader_LHG = get_class_loaders(
        TrainDataset=TrainDataLoader,
        TestDataset=TestDataLoader,
        transforms=val_transforms,
        fold=fold,
        loader_id='LHG',
        pkl_path=pkl_paths[2]
    )

    # Move model to GPU if available
    if config.use_cuda:
        model.cuda()

    cross_entropy_loss = CrossEntropyLoss(reduction='mean')  # Define cross-entropy loss

    # Only optimize parameters that require gradients
    optim_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        optim_parameters,
        lr=config.lr_self,
        weight_decay=config.weight_decay
    )  # Using Adam optimizer

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        factor=config.lr_reduce_factor,
        patience=config.lr_reduce_patience,
        mode='max',
        min_lr=config.min_lr
    )

    global_step = 0  # Initialize global step
    best_score = -1  # Initialize best score

    # Initialize DataFrames for logging
    log_1p19q = pd.DataFrame(columns=[
        'epoch',
        'Train_loss', 'Train_Accuracy', 'Train_F1',
        'Val_loss', 'Val_Accuracy', 'Val_F1'
    ])

    log_IDH = pd.DataFrame(columns=[
        'epoch',
        'Train_loss', 'Train_Accuracy', 'Train_F1',
        'Val_loss', 'Val_Accuracy', 'Val_F1'
    ])

    log_LHG = pd.DataFrame(columns=[
        'epoch',
        'Train_loss', 'Train_Accuracy', 'Train_F1',
        'Val_loss', 'Val_Accuracy', 'Val_F1'
    ])

    log_mean = pd.DataFrame(columns=[
        'epoch',
        'Train_loss', 'Train_Accuracy', 'Train_F1',
        'Val_loss', 'Val_Accuracy', 'Val_F1'
    ])

    model.train()  # Set model to training mode

    # Initialize iterator indices
    iter_index_IDH = 0
    iter_index_LHG = 0

    iter_IDH = iter(train_loader_IDH)
    iter_LHG = iter(train_loader_LHG)

    for epoch in range(config.epochs_s1):
        if epoch == config.save_epoch:
            best_score = -1  # Reset best score at a specific epoch

        iter_1p19q = iter(train_loader_1p19q)  # Get iterator for 1p19q class
        for batch_idx in range(len(train_loader_1p19q)):
            iter_index_IDH += 1
            iter_index_LHG += 1

            # Reset iterators if they exceed dataset length
            if iter_index_IDH >= len(train_loader_IDH):
                iter_IDH = iter(train_loader_IDH)
                iter_index_IDH = 0

            if iter_index_LHG >= len(train_loader_LHG):
                iter_LHG = iter(train_loader_LHG)
                iter_index_LHG = 0

            # Get data for each class
            data_1p19q = next(iter_1p19q)
            data_IDH = next(iter_IDH)
            data_LHG = next(iter_LHG)

            # Reshape input data
            imgs_1p19q, label_NT_1p19q, label_1p19q = reshape_input_data(data_1p19q)
            imgs_IDH, label_NT_IDH, label_IDH = reshape_input_data(data_IDH)
            imgs_LHG, label_NT_LHG, label_LHG = reshape_input_data(data_LHG)

            # Combine labels and inputs
            label_list = [label_1p19q, label_IDH, label_LHG]
            combined_input = torch.cat((imgs_1p19q, imgs_IDH, imgs_LHG), dim=0)
            labels_NG_list = [label_NT_1p19q, label_NT_IDH, label_NT_LHG]

            # Forward pass through the model
            (logits_NG, labels_NG_all, out_cls_list_NG,
             out_cls_list, out_cls_labels) = model(combined_input, labels_NG_list, label_list)

            # Compute losses
            loss_NG = cross_entropy_loss(logits_NG, labels_NG_all)
            loss_cls_N1 = cross_entropy_loss(out_cls_list_NG[0], label_list[0])
            loss_cls_N2 = cross_entropy_loss(out_cls_list_NG[1], label_list[1])
            loss_cls_N3 = cross_entropy_loss(out_cls_list_NG[2], label_list[2])
            loss_cls_1 = cross_entropy_loss(out_cls_list[0], out_cls_labels[0])
            loss_cls_2 = cross_entropy_loss(out_cls_list[1], out_cls_labels[1])
            loss_cls_3 = cross_entropy_loss(out_cls_list[2], out_cls_labels[2])

            # Total classification loss
            total_cls_loss = loss_cls_1 + loss_cls_2 + loss_cls_3 + loss_cls_N1 + loss_cls_N2 + loss_cls_N3
            # Total loss
            total_loss = loss_NG + total_cls_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_step += 1

        # Get current learning rate
        current_lr = get_learning_rate(optimizer)
        print(f'Global_Step {global_step} | Train Epoch: {epoch} | lr {current_lr:.2e} |')

        # Define save path
        save_path = os.path.join(config.data_dir, 'results', str(save_name), f'Fold_{fold}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Compute metrics for each class
        metrics_train_1p19q, metrics_test_1p19q = valid_metrics_by_class(
            model=model,
            train_loader=val_train_loader_1p19q,
            test_loader=val_loader_1p19q,
            ssl_index=0,
            class_name='1p19q'
        )

        metrics_train_IDH, metrics_test_IDH = valid_metrics_by_class(
            model=model,
            train_loader=val_train_loader_IDH,
            test_loader=val_loader_IDH,
            ssl_index=1,
            class_name='IDH'
        )

        metrics_train_LHG, metrics_test_LHG = valid_metrics_by_class(
            model=model,
            train_loader=val_train_loader_LHG,
            test_loader=val_loader_LHG,
            ssl_index=2,
            class_name='LHG'
        )

        # Calculate mean metrics across classes
        metrics_train_mean = (np.array(metrics_train_1p19q) + np.array(metrics_train_IDH) +
                              np.array(metrics_train_LHG)) / 3.0
        metrics_test_mean = (np.array(metrics_test_1p19q) + np.array(metrics_test_IDH) +
                             np.array(metrics_test_LHG)) / 3.0

        # Create log entries
        log_entry_1p19q = get_pandas_series(epoch, metrics_train_1p19q, metrics_test_1p19q)
        log_entry_IDH = get_pandas_series(epoch, metrics_train_IDH, metrics_test_IDH)
        log_entry_LHG = get_pandas_series(epoch, metrics_train_LHG, metrics_test_LHG)
        log_entry_mean = get_pandas_series(epoch, metrics_train_mean, metrics_test_mean)

        # Append logs to DataFrames and save to CSV
        log_1p19q = log_1p19q.append(log_entry_1p19q, ignore_index=True)
        log_1p19q.to_csv(os.path.join(save_path, 'log_1p19q.csv'), index=False)

        log_IDH = log_IDH.append(log_entry_IDH, ignore_index=True)
        log_IDH.to_csv(os.path.join(save_path, 'log_IDH.csv'), index=False)

        log_LHG = log_LHG.append(log_entry_LHG, ignore_index=True)
        log_LHG.to_csv(os.path.join(save_path, 'log_LHG.csv'), index=False)

        log_mean = log_mean.append(log_entry_mean, ignore_index=True)
        log_mean.to_csv(os.path.join(save_path, 'log_mean.csv'), index=False)

        # Use the mean F1 score from the test set as the evaluation metric
        current_f1 = metrics_test_mean[2]

        # Save the best model based on F1 score
        if epoch >= config.save_epoch:
            if current_f1 > best_score:
                save_config = {
                    'name': save_name,
                    'save_dir': save_path,
                    'global_step': global_step,
                    'Test_F1': current_f1
                }
                save_model(model=model, config=save_config)
                best_score = current_f1

        # Update the best score if current F1 is better
        if current_f1 > best_score:
            best_score = current_f1

        # Step the scheduler based on the best score
        scheduler.step(best_score)


if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")

    # Set random seeds for reproducibility
    seed = config.random_seed
    np.random.seed(seed)  # Python's random seed
    os.environ['PYTHONHASHSEED'] = str(seed)  # Disable Python hash randomization
    torch.manual_seed(seed)  # Torch CPU random seed
    torch.cuda.manual_seed(seed)  # Torch GPU random seed
    torch.cuda.manual_seed_all(seed)  # Torch multi-GPU random seed
    torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmark
    torch.backends.cudnn.deterministic = True  # Enable deterministic algorithms

    # Define paths to pickle files for each class
    pkl_paths = [
        './1p19q.pkl',
        './IDH.pkl',
        './LHG.pkl'
    ]

    # Perform 5-fold cross-validation training
    for fold in range(5):
        model = CFE(in_channels=3, num_classes=2, Arcm=0.5, multi=3)  # Initialize the model
        save_name = model.name  # Get the model's name
        print(f"{save_name}_Fold_{fold}")
        train(fold, pkl_paths)  # Start training