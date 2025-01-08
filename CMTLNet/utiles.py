import os
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def save_model(model, config):
    """
    Saves the given model to the specified directory with a filename that includes
    the model's name, F1 score, and training step.

    Args:
        model (torch.nn.Module): The model to be saved.
        config (dict): Configuration dictionary containing:
            - 'Test_F1' (float): The F1 score from testing.
            - 'name' (str): The base name for the model file.
            - 'global_step' (int): The current training step.
            - 'save_dir' (str): Directory where the model will be saved.
    """
    # Convert F1 score to percentage
    f1_percentage = config['Test_F1'] * 100

    # Create a filename with model name, F1 score, and training step
    filename = "{}_F1_{:.2f}_step_{}.pkl".format(
        config['name'],
        f1_percentage,
        config['global_step']
    )

    # Construct the full path for saving the model
    model_save_path = os.path.join(config['save_dir'], filename)

    # Save the model using PyTorch's save function
    torch.save(model, model_save_path)
    print("Saved model to {}".format(model_save_path))


def get_learning_rate(optimizer):
    """
    Retrieves the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer from which to extract the learning rate.

    Returns:
        float: The current learning rate.

    Raises:
        ValueError: If the optimizer has no parameter groups.
    """
    if len(optimizer.param_groups) > 0:
        # Return the learning rate of the first parameter group
        return optimizer.param_groups[0]['lr']
    else:
        raise ValueError('No trainable parameters found in the optimizer.')


def move_tensor_to_device(tensor, use_gpu=False):
    """
    Moves a tensor to GPU or CPU based on the specified flag.

    Args:
        tensor (torch.Tensor): The tensor to move.
        use_gpu (bool, optional): If True, move the tensor to GPU. Otherwise, move to CPU. Defaults to False.

    Returns:
        torch.Tensor: The tensor on the specified device.
    """
    if use_gpu:
        # Move tensor to GPU asynchronously
        return tensor.cuda(non_blocking=True)
    else:
        # Move tensor to CPU
        return tensor.cpu()


def calculate_classification_metrics(predictions, targets, average_method='macro'):
    """
    Calculates various classification metrics based on predictions and true targets.

    Args:
        predictions (array-like): Predicted class labels.
        targets (array-like): True class labels.
        average_method (str, optional): Method to average the metrics. Defaults to 'macro'.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): Accuracy score.
            - f1 (float): F1 score.
            - precision (float): Precision score.
            - recall (float): Recall score.
            - confusion_mat (numpy.ndarray): Confusion matrix.
    """
    # Compute the confusion matrix
    confusion_mat = confusion_matrix(targets, predictions)
    confusion_mat = confusion_mat.astype('int')

    # Calculate F1 score
    f1 = f1_score(targets, predictions, average=average_method)

    # Calculate precision score
    precision = precision_score(targets, predictions, average=average_method)

    # Calculate recall score
    recall = recall_score(targets, predictions, average=average_method)

    # Calculate accuracy score
    accuracy = accuracy_score(targets, predictions)

    return accuracy, f1, precision, recall, confusion_mat


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking metrics like loss or accuracy during training.
    """

    def __init__(self):
        """
        Initializes the AverageMeter by resetting all metrics.
        """
        self.reset()

    def reset(self):
        """
        Resets all metrics to their initial state.
        """
        self.current = 0
        self.average = 0
        self.total = 0
        self.count = 0

    def update(self, value, n=1):
        """
        Updates the metrics with a new value.

        Args:
            value (float): The new value to incorporate.
            n (int, optional): The number of occurrences of the value. Defaults to 1.
        """
        self.current = value
        self.total += value * n
        self.count += n
        self.average = self.total / self.count