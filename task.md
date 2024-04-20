# Task for ML Interview

Your task is to develop a machine learning approach to predict the subjects of scientific papers.

## Dataset
The Cora dataset consists of 2708 scientific publications classified into one of seven classes (`Case_Based`, `Genetic_Algorithms`, `Neural_Networks`, `Probabilistic_Methods`, `Reinforcement_Learning`, `Rule_Learning`, `Theory`). The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download Link: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

## Task
Your task is to develop a machine learning approach to predict the subjects of scientific papers:

1. Load the data
2. Split the dataset using 10-fold cross validation
3. Develop a machine learning approach to learn and predict the subjects of papers
4. Store your predictions in a file as tab-separated values (TSV) in the format `<paper_id> <class_label>` where *class_label* is a string.
5. Evaluate your approach in terms of *accuracy* indicating the percentage of nodes that were predicted correctly.

## Organization

0. Create a GitHub repository, grant access to the GitHub users `heindorf` and send an email with the name of the repository to `heindorf@uni-paderborn.de`.
1. Upload your code (Python preferred) to the repository
2. Upload your predictions (TSV file) to the repository
3. Document your approach in the README file. Give an overview of your approach and describe how your approach can be executed. Ideally, it should be possible to execute your approach with a single command.
