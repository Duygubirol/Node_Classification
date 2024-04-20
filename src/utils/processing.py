import torch
import numpy as np
import scipy.sparse as sp
import argparse
from collections import Counter
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def load_data(path="./data/cora/", dataset="cora"):
    """
    Load the Cora dataset which consists of paper IDs, features, and labels, and prepare it for graph-based learning models.

    Parameters:
    - path (str): The path to the dataset directory. Default is "./data/cora/".
    - dataset (str): The name of the dataset. Default is "cora".

    Returns:
    - adj (Tensor): The adjacency matrix of the graph in sparse tensor format.
    - features (Tensor): Feature matrix as a FloatTensor.
    - labels (Tensor): Label tensor as a LongTensor.
    - idx_train (Tensor): Indices of the training samples.
    - idx_val (Tensor): Indices of the validation samples.
    - idx_test (Tensor): Indices of the test samples.
    - paper_ids (np.ndarray): Array of paper IDs.
    - inverted_label_mapping (dict): Dictionary mapping label indices to string labels.
    """
    print('Loading {} dataset...'.format(dataset))

    # Load data from the Cora dataset content file, which includes paper IDs, features, and class labels.
    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str))

    # Extract paper IDs which are unique identifiers for each paper.
    paper_ids = idx_features_labels[:, 0]

    # Features are all columns after the ID and before the last column, converted to a sparse matrix format.
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # The last column contains labels; extract these for further processing.
    label_strings = idx_features_labels[:, -1]

    # Create a mapping from label names to a unique integer index.
    label_mapping = create_label_mapping(label_strings)

    # Encode labels using one-hot encoding based on the label mapping.
    labels = encode_onehot(label_strings, label_mapping)

    # Invert the label mapping to enable easy lookup from index to label name.
    inverted_label_mapping = {v: k for k, v in label_mapping.items()}
    # print(inverted_label_mapping)

    # Load the citation links, which represent the graph structure as an edge list.
    idx = np.array(paper_ids, dtype=np.int32)  # Reuse paper_ids
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}{}.cites".format(path, dataset), dtype=np.int32)

    # Map the citation links to indices in the features matrix.
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # Create the adjacency matrix for the graph where each link is represented by a 1 (symmetric matrix).
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize feature matrix and adjacency matrix to facilitate learning.
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # Extract label indices
    label_ids = np.argmax(labels, axis=1)

    # Split the data into training, validation, and test sets.
    idx_train_val, idx_test, y_train_val, y_test, paper_ids_train_val, paper_ids_test = train_test_split(
        range(len(labels)), label_ids, paper_ids, test_size=0.13, random_state=42, stratify=label_ids)
    idx_train, idx_val, y_train, y_val, paper_ids_train, paper_ids_val = train_test_split(
        idx_train_val, y_train_val, paper_ids_train_val, test_size=0.13, random_state=42, stratify=y_train_val)

    # Convert feature, label, and adjacency matrices from Numpy arrays to PyTorch tensors for model training.
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # Convert indices to tensors for indexing during training.
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print("Train set size:", len(idx_train))
    print("Validation set size:", len(idx_val))
    print("Test set size:", len(idx_test))

    return adj, features, labels, idx_train, idx_val, idx_test, paper_ids_test, inverted_label_mapping


def create_label_mapping(labels):
    """
    Creates a dictionary mapping from labels to indices based on the unique labels found in the dataset.

    Parameters:
    - labels (np.ndarray): Array of label strings.

    Returns:
    - dict: A dictionary where each label string is mapped to a unique integer.
    """
    unique_labels = np.unique(labels)
    return {label: idx for idx, label in enumerate(unique_labels)}


def encode_onehot(labels, label_mapping):
    """
    Encodes labels into a one-hot numpy array using a predefined label mapping.

    Parameters:
    - labels (list[str]): List of label strings.
    - label_mapping (dict): Dictionary mapping labels to indices.

    Returns:
    - np.ndarray: One-hot encoded representation of labels as a numpy array.
    """
    num_classes = len(label_mapping)
    labels_onehot = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, label in enumerate(labels):
        label_index = label_mapping[label]  # Directly use label to get index
        labels_onehot[i, label_index] = 1
    return labels_onehot


def normalize(mx):
    """
    Normalize a sparse matrix row-wise.

    Parameters:
    - mx (sp.spmatrix): A scipy sparse matrix.

    Returns:
    - sp.spmatrix: Row-normalized sparse matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix into a Torch sparse tensor.

    Parameters:
    - sparse_mx (sp.spmatrix): A scipy sparse matrix.

    Returns:
    - torch.sparse.FloatTensor: The input matrix as a PyTorch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_arguments():
    """
    Parse command line arguments for the training script. This function allows customization of the training process through various
    command line options such as disabling CUDA, setting training parameters like learning rate and epochs, and others. Defaults are 
    set based on preliminary guidelines from relevant literature, with further adjustments planned based on additional articles.

    Note:The choice of default parameter values is obtained from this reference article: https://doi.org/10.48550/arXiv.1609.02907 
    
    Returns:
        argparse.Namespace: An object containing the command line arguments and their values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true',
                        default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    return parser.parse_args()


def print_class_distribution(labels, idx, description):
    """
    Prints the distribution of classes for a subset of data specified by indices.

    This function is used to display the number of samples for each class in a given subset, helping to visualize the balance or imbalance among classes. Useful for understanding dataset characteristics or the results of data manipulation like sampling.

    Parameters:
    - labels (np.ndarray or torch.Tensor): Array or tensor of label data.
    - idx (np.ndarray or torch.Tensor): Indices of the samples to be considered for printing the distribution.
    - description (str): Descriptive text to precede the distribution output, e.g., 'Training' or 'Validation'.

    Output:
    - Prints the distribution of classes to the standard output.
    """
    label_counts = Counter(labels[idx].cpu().numpy())
    print(f"{description} data class distribution:")
    for class_label in sorted(label_counts):
        print(f"Class {class_label}: {label_counts[class_label]} samples")


def balance_classes(args, features, labels, idx):
    """
    Balances the classes in the dataset by undersampling to the smallest class's size among the specified indices.

    This function helps in creating a balanced training set by undersampling all classes to the size of the smallest class in the dataset subset specified by `idx`. This can mitigate issues related to class imbalance which might affect model training.

    Parameters:
    - features (np.ndarray or torch.Tensor): Array or tensor of feature data. Only used to validate alignment with labels, not modified by the function.
    - labels (np.ndarray or torch.Tensor): Array or tensor of label data corresponding to all samples.
    - idx (np.ndarray): Indices representing the subset of data to be balanced.

    Returns:
    - np.ndarray: New indices representing a balanced subset of the original indices.

    Note:
    - The function does not modify the input features and labels, it only resamples the indices.
    """
    unique_classes = np.unique(labels[idx])
    min_size = min(Counter(labels[idx]).values())  # size of the smallest class
    new_indices = np.array([], dtype=int)

    for cls in unique_classes:
        cls_indices = idx[labels[idx] == cls]
        undersampled_indices = resample(
            cls_indices, replace=False, n_samples=min_size, random_state=args.seed)
        new_indices = np.concatenate([new_indices, undersampled_indices])

    np.random.shuffle(new_indices)  # Shuffle to mix class samples
    return new_indices
