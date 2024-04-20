import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from src.model.models import GCN
from src.utils import (load_data,
                       plot_fold_performance,
                       confusion_matrix_plot,
                       parse_arguments,
                       print_class_distribution,
                       balance_classes)


def train():
    """
    Conducts the training and evaluation of a Graph Convolutional Network (GCN) for node classification using cross-validation,
    followed by testing the best-performing model on a separate test dataset. This process includes configuration and environment
    setup, data loading, executing training across multiple folds to ensure model robustness, and evaluating model performance on
    unseen data.

    Training involves configuring GPU usage (if available), setting random seeds for reproducibility, loading graph data,
    initializing the GCN model with specified hyperparameters, setting up directories for saving model checkpoints and reports,
    and executing a training loop for each fold. Each fold includes class distribution balancing, multiple training epochs with
    model weight updates via backpropagation, and model validation. Metrics are recorded and the best model is saved.

    After training, the best model is evaluated on a test dataset to gauge performance on new data, and final metrics are reported.
    Results, including model predictions and performance plots, are saved for analysis.

    Returns:
        It includes saved models, performance plots, and terminal logs of training/validation metrics and settings, as well
        as test evaluation results).
    """

    ###################################
    # Configuration and Data Loading
    ###################################

    args = parse_arguments()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load the data
    adj, features, labels, idx_train, idx_val, idx_test, paper_ids_test, inverted_label_mapping = load_data()

    # Combine train and validation indices to apply apply 10 fold cross validation
    combined_idx = np.concatenate((idx_train, idx_val))

    # Initialize the model
    model = GCN(nfeat=features.shape[1], nhid=args.hidden,
                nclass=labels.max().item() + 1, dropout=args.dropout)

    # Move model to CUDA if available
    if args.cuda:
        model = model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Ensure the directory for saving models and reports exists
    model_dir = 'saved_models'
    report_dir = 'reports'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Cross-validation setup
    num_folds = 10
    skf = StratifiedKFold(n_splits=num_folds,
                          shuffle=True, random_state=args.seed)
    best_val_acc = 0
    best_fold = None
    best_model_path = None
    all_fold_train_accs = []
    all_fold_val_accs = []

    ###########################
    # Training Loop
    ###########################

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(combined_idx)), labels[combined_idx].numpy())):
        print(f"Fold {fold + 1}")
        balanced_train_idx = balance_classes(
            args, features, labels.numpy(), train_idx)
        train_idx = torch.LongTensor(balanced_train_idx)
        val_idx = torch.LongTensor(val_idx)

        # Training and validation index preparation
        if args.cuda:
            train_idx = train_idx.cuda()
            val_idx = val_idx.cuda()

        # Instantiate model and optimizer for the fold
        model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max(
        ).item() + 1, dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        all_pred_vals = []
        all_true_vals = []

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[train_idx], labels[train_idx])
            loss_train.backward()
            optimizer.step()

            # Evaluation on validation set
            model.eval()
            output = model(features, adj)
            pred_train = output[train_idx].max(1)[1]
            train_acc = (pred_train == labels[train_idx]).float().mean().item()
            pred_val = output[val_idx].max(1)[1]
            val_acc = (pred_val == labels[val_idx]).float().mean().item()

            train_losses.append(loss_train.item())
            val_losses.append(F.nll_loss(
                output[val_idx], labels[val_idx]).item())
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            # Store validation predictions
            all_pred_vals.extend(pred_val.cpu().numpy())
            # Store true labels
            all_true_vals.extend(labels[val_idx].cpu().numpy())

        # Call the plot function after the fold training and validation are complete
        plot_fold_performance(train_losses, val_losses, train_accs,
                              val_accs, fold, report_dir)

        all_fold_train_accs.append(train_accs[-1])
        all_fold_val_accs.append(val_accs[-1])

        # Update best model if current validation is higher
        if max(val_accs) > best_val_acc:
            best_val_acc = max(val_accs)
            best_fold = fold
            best_model_path = os.path.join(model_dir, f'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

        # Print class distribution and accuracies for the current fold
        print_class_distribution(
            labels, train_idx, f"Fold {fold + 1} - Training")
        print(f"Training Accuracy: {train_accs[-1]:.4f}")
        print_class_distribution(
            labels, val_idx, f"Fold {fold + 1} - Validation")
        print(f"Validation Accuracy: {max(val_accs):.4f}")

    # Calculate mean and standard deviation for training and validation accuracies across all folds
    mean_train_acc = np.mean(all_fold_train_accs)
    std_train_acc = np.std(all_fold_train_accs)
    mean_val_acc = np.mean(all_fold_val_accs)
    std_val_acc = np.std(all_fold_val_accs)

    # Print out the overall training and validation statistics
    print(
        f"Overall Mean Training Accuracy: {mean_train_acc:.4f} ± {std_train_acc:.4f}")
    print(
        f"Overall Mean Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")

    ###########################
    # Testing Loop
    ###########################

    # Load best model to evaluate on test set
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        output = model(features, adj)
        pred_test = output[idx_test].max(1)[1]
        test_acc = (pred_test == labels[idx_test]).float().mean().item()
        # Print the label mapping here to check its contents just before it's accessed
        print(inverted_label_mapping)

        # Check Data Size
        if len(idx_test) != len(labels[idx_test]):
            print(
                "Warning: Sizes of idx_test and labels[idx_test] do not match!")

        # Calculate all_true_vals_test
        all_true_vals_test = labels[idx_test].cpu().numpy()

        # Print final class distribution and test accuracy
        print_class_distribution(labels, idx_test, "Final Test")
        print(f"Best Fold: {best_fold + 1}")
        print(f"Test Accuracy with Best Model: {test_acc:.4f}")
        confusion_matrix_plot(all_true_vals_test, pred_test, report_dir)

        # Create result directory if not exists
        os.makedirs('result', exist_ok=True)

    with open('result/predictions.tsv', 'w') as f:
        f.write("paper_id\tpred_class_label\ttrue_class_label\n")
        for paper_id, pred_label, true_label in zip(paper_ids_test, pred_test, labels[idx_test]):
            # Using inverted mapping
            pred_label_str = inverted_label_mapping[pred_label.item()]
            true_label_str = inverted_label_mapping[true_label.item()]
            f.write(f"{paper_id}\t{pred_label_str}\t{true_label_str}\n")


# Entry point of the script
if __name__ == '__main__':
    train()
