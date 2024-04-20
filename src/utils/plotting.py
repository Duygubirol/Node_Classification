import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_fold_performance(train_losses, val_losses, train_accs, val_accs, fold, report_dir):
    """
    Plot training and validation performance metrics including losses, accuracies, and a confusion matrix for a specific fold of model evaluation.

    This function generates two subplots: one for losses over epochs and one for accuracies over epochs. It then saves the plot to a specified directory.

    Parameters:
    - train_losses (list of float): List of training loss values over epochs.
    - val_losses (list of float): List of validation loss values over epochs.
    - train_accs (list of float): List of training accuracy values over epochs.
    - val_accs (list of float): List of validation accuracy values over epochs.
    - fold (int): The current fold number (zero-indexed) for which the metrics are being plotted.
    - report_dir (str): Directory path where the plot image will be saved.

    Side Effects:
    - Creates a file `fold_{fold + 1}_plot.png` in the specified `report_dir` containing the generated plots.
    """
    # Increase the width to accommodate three subplots
    plt.figure(figsize=(10, 5))

    # Plot for training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} Losses')
    plt.legend()

    # Plot for training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Fold {fold + 1} Accuracies')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{report_dir}/fold_{fold + 1}_plot.png')
    plt.close()


def confusion_matrix_plot(all_true_vals_test, pred_test, report_dir):
    """
    Plot the confusion matrix for the test set.

    This function generates a confusion matrix plot based on the true and predicted labels of the test set. It then saves the plot to a specified directory.

    Parameters:
    - all_true_vals_test (list): List of true label values for the test set.
    - pred_test (tensor): Tensor containing python trthe predicted label values for the test set.
    - report_dir (str): Directory path where the plot image will be saved.

    Side Effects:
    - Creates a file `test_confusionmatrix.png` in the specified `report_dir` containing the confusion matrix plot.
    """
    # Plot for the confusion matrix
    cm = confusion_matrix(all_true_vals_test, pred_test.cpu().numpy())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.tight_layout()
    plt.savefig(f'{report_dir}/test_confusionmatrix.png')
    plt.close()
