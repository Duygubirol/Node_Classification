# Node Classification with  Graph Convolution Network (GCN) on Cora Dataset
## Stratified Split with 10-fold cross validation 

This repository contains the implementation of Graph Convolutional Networks (GCN) in PyTorch for node classification on Cora dataset, developed by Duygu Ekinci Birol.
It includes a training of a model to predict the class or label of a node, commonly known as node classification. 


### Installation
Before installing the package, ensure you are using **Python 3.10**. This is necessary to maintain compatibility with the project's dependencies.

#### Steps to Install
To install the package and its dependencies, you can use the following steps:

1. Clone this repository:
```bash
git clone https://github.com/Duygubirol/Node_Classification
```

2. Navigate to the project directory:
```bash
cd Node_Classification
```

3. Set up a Python environment (recommended):
Creating a virtual environment is recommended to avoid conflicts with other projects or system packages.
```bash
conda create -n py310 python=3.10 -y
conda activate py310
```

4. Upgrade pip:
Ensure that pip is up-to-date to avoid any issues during the installation of dependencies.
for windows:
```bash
python -m pip install --upgrade pip
```
for mac:
```bash
pip install --upgrade pip
```

4. Install the package:
This step installs the package along with all its Python dependencies as specified in the setup.py file.
```bash
pip install .
```

Alternatively, dependencies can be installed using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Dependencies

The project's dependencies are managed using `setup.py`. The required libraries and their versions are specified in the `install_requires` parameter of the `setup()` function in the `setup.py` file.

For more information on the dependencies, please refer to the `setup.py` file in the root directory of this repository.


### Usage
The `train.py` script facilitates the training and evaluation of the GCN model for the node classification task on the Cora dataset. It handles the stratified split with 10-fold cross-validation, ensuring a robust assessment of the model's performance.

Running the script initiates a sequence of steps:
- Data loading and processing,
- Graph construction,
- GCN model instantiation,
- Model training with validation,
- And finally, evaluation on the test dataset.

To begin training and classification, execute:
```bash
python train.py
```

## File Structure and Description
<details>
<summary>Detail information</summary>

1. **train.py**: This script facilitates the training and evaluation of the GCN model for node classification on the Cora dataset. It handles data loading, graph construction, model instantiation, training with validation, and evaluation on the test dataset.
2. **models.py**: This file contains the definition of the Graph Convolutional Network (GCN) model used for node classification. It includes the architecture of the GCN, including the graph convolutional layers and the forward pass logic.
3. **layer.py**: This file contains a GraphConvolution class that implements a graph convolution layer for neural networks, which applies a weight matrix to input features and propagates them using a graph's adjacency matrix.
3. **processing.py**: This module contains utility functions used throughout the project, such as data loading functions, data preprocessing functions, and helper functions for training and evaluation.
4. **plotting.py**: This module contains functions for plotting performance metrics during training, such as loss curves, accuracy curves, and confusion matrices. It helps visualize the model's performance during training and evaluation.
5. **README.md** : This file provides an overview of the project, installation instructions, usage instructions, dependencies, and other relevant information for users and contributors.
6. **requirements.txt**: This file lists all the Python dependencies required to run the project. Users can install these dependencies using pip install -r requirements.txt.
7. **saved_models/**: This directory contains saved model weights obtained after training the GCN model. These weights can be loaded to reproduce or continue training from a specific checkpoint.
8. **reports/**: This directory contains performance plots generated during training and evaluation, such as loss curves, accuracy curves, and confusion matrices. These plots help analyze the model's performance and identify any issues during training.
9. **result/**: This directory contains result files generated after evaluating the model on the test dataset, such as predictions.tsv. These files provide insights into the model's predictions and can be used for further analysis or reporting.

</details>

## Overview of the Approach:

### Data
<details>
<summary>Detail information</summary>

The Cora dataset, which can be downloaded from the provided link (https://linqs-data.soe.ucsc.edu/public/lbc/cora.), is a collection of academic publications represented as nodes in a graphintroduced by Yang et al. (2016) [1]. Each publication is connected to others through citations, forming links or edges between them. Because all nodes in the Cora dataset represent the same type of entity (research papers) and all edges represent the same type of relationship (citations), the dataset is considered homogeneous. This homogeneity simplifies the modeling process and allows for the application of techniques like graph convolutional networks (GCNs) to analyze and make predictions on the dataset.

The nodes in the Cora dataset are categorized into seven subjects: "Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods", "Reinforcement_Learning", "Rule_Learning", and "Theory". These subjects represent different fields or topics of study within academia.

The goal of using this dataset is to train a model to predict the subject of each publication based on its features and its relationships with other publications. In other words, given the content of a publication and the publications it cites or is cited by, the model aims to classify it into one of the seven predefined subjects.
There are 3 main training approach in terms of data splitting for this dataset:
1. Fixed 20 nodes per class: public split for benchmarking [1],[2]
2. Random splits [2]
3. The ML models are trained on papers up to 2017, validated on those from 2018, and tested on papers published since 2019. This facilitates their application in real-world scenarios, like assisting ARXIV moderators [3]

In this repository, data splitting was conducted based on a stratified distribution of labels in the general dataset for training, validation, and testing. Since the task requires k-fold cross-validation, using fixed nodes for training is not feasible. Therefore, this repository is not intended for benchmarking comparisons. However, the training data during cross-validation is balanced, as described in these articles [1],[2], while validation and test data retained the same distribution as the dataset. To assess the dataset's generalization capability after 10 rounds of validation, the best model was evaluated on the test dataset. Hence, the approach employed random stratified splitting with balanced training data in this repository. An alternative approach could involve applying the third method.

</details>

### Model
<details>
<summary>Detail information</summary>

![Multi-layer GCN with first-order filters [2]](readme/GCN.png)

This repository utilizes a GCN [2], a type of graph neural network designed for processing and learning from graph-structured data. At its core, the GCN employs a "graph convolution" layer, which operates similarly to a conventional dense layer but incorporates information from the adjacency matrix of the graph and the feature matrix of the nodes as inputs.This allows the model to consider a node's connections when making predictions or learning representations.

**Architecture of GCNs**: A GCN layer comprises trainable parameters (weight matrix and bias vector), input features (node features matrix), and the normalized graph adjacency matrix. During operation, the layer applies graph convolution to smooth node feature vectors based on their neighboring nodes in the graph, facilitating information propagation and feature learning.

**Working of GCNs**: During training, GCNs iteratively aggregate information from neighboring nodes and update node representations accordingly. By traversing through multiple layers of GCNs, the model captures hierarchical representations of nodes, encompassing both local and global graph structures.

**Reasons for Choosing the GCN Model**:  According to recent research, simple GNN architectures can effectively solve problems at this small scale[4],[5], and the performance across different GNNs on these datasets is almost indistinguishable [6]. The GCN model was selected based on its demonstrated superior performance compared to alternative methods, particularly in terms of classification accuracy and computational efficiency. Its layer-wise propagation rule, inspired by spectral graph convolutions, enables efficient and scalable semi-supervised classification on graphs. Additionally, by leveraging the graph structure directly within the neural network architecture, the GCN achieves high classification accuracy while maintaining computational efficiency. 

</details>

### Node Classification
<details>
<summary>Detail information</summary>
Node classification is a task in graph-based machine learning where each node in a graph is assigned a label or category based on its features and its connections to other nodes. 
A GCN takes the adjacency matrix of the graph (which encodes the connections between nodes) and the node features (like text attributes or numerical data) as inputs:

- **Graph Convolution**: At each layer of a GCN, a node's features are updated by aggregating and transforming features from its neighboring nodes. This aggregation mechanism allows the model to propagate and integrate label information across the network, enabling the classification of nodes even when labels are available only for a small fraction of the entire graph.

- **Layer-wise Learning**: Multiple layers in GCNs allow the capture of not just immediate neighbor relationships but also more extended neighborhoods. This helps in understanding deeper contextual relationships within the graph, essential for accurate classification.

#### Training the GCN Model

The training involves:
- **Semi-supervised Learning**: Only a subset of nodes have labels during training. The challenge is to effectively use this limited labeled data along with the graph structure to predict labels for all nodes.
- **Cross-validation**: This process helps in validating the model's effectiveness and ensuring it generalizes well over different parts of the graph.
- **Loss Calculation and Backpropagation**: The loss function typically measures the difference between the predicted labels and the actual labels of the trained nodes. The model's parameters are optimized by minimizing this loss over several training iterations.

</details>

### Results
<details>
<summary>Detail information</summary>
After training and validating the GCN across 10 folds of cross-validation on the Cora dataset, the model demonstrated robust performance. Below are the summarized outcomes of the model's accuracy:

Training and Validation Performance:

**Mean Training Accuracy:** 0.9272 ± 0.0072

**Mean Validation Accuracy:**0.8637 ± 0.0211

These results represent the average performance across all 10 folds, showcasing the model's ability to generalize across different subsets of the data.

**Final Model Evaluation with the best model:**

Test Accuracy: 0.8782

The test accuracy indicates the effectiveness of the model when applied to unseen data, providing a realistic measure of its predictive power in practical scenarios.
This approach not only adds valuable insights about the model's 

</details>

## Citations

Yang, Z., Cohen, W. W., & Salakhutdinov, R. (2016). Revisiting semi-supervised learning with graph embeddings. In Proceedings of the 33rd International Conference on Machine Learning (Vol. 48). New York, NY, USA: JMLR: Workshop and Conference Proceedings[1]

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. Proceedings of the International Conference on Learning Representations (ICLR). https://doi.org/10.48550/arXiv.1609.02907 [2].

Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (Year). Open Graph Benchmark: Datasets for Machine Learning on Graphs. Department of Computer Science, Stanford University.[3]. 

Wu, Y., Lian, D., Xu, Y., Wu, L., & Chen, E. (2020). Graph Convolutional Networks with Markov Random Field Reasoning for Social Spammer Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 34(01), 5455. https://doi.org/10.1609/aaai.v34i01.5455 [4]. 

Bojchevski, A., Shchur, O., Zügner, D., & Günnemann, S. (2018). NetGAN: Generating Graphs via Random Walks. In Proceedings of the 35th International Conference on Machine Learning (Vol. 80, pp. 610-619). PMLR [5]. 

Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2023). Benchmarking Graph Neural Networks. Journal of Machine Learning Research, 24(43), 1-48 [6].
