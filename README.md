# Machine Learning Project
## Univeristy of Cagliari - Applied Computer Science and Data Analytics course

## Index
- [Introduction](#Introduction)
- [Implementation](#Implementation)
- [Installation](#Installation - Jupyter Notebook) 

### Introduction
This project involves the analysis and exploration of various Machine Learning techniques applied to classification problems. The work will focus on a specific dataset: the [**Dry Bean Dataset**](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) from the [**UCI Machine Learning Repository**](https://archive.ics.uci.edu/). The project aims to implement and compare different classification algorithms to identify bean varieties based on their morphological characteristics.

### Implementation
The project will implement several machine learning algorithms, including:

- [**Decision Tree**](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [**K-Nearest Neighbors**](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [**Naive Bayes (GaussianNB)**](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [**Support Vector Classifier**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [**Random Forest**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [**Voting Classifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)

Additionally, hyperparameter tuning will be performed using [**GridSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and cross-validation with [**StratifiedKFold**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

The various results will be compared to identify the model that achieves the highest accuracy on the dry bean classification task.

Please note that all analyses performed on the dataset (in the jupyter notebook) are saved as .png files in dedicated folders for easy reference and documentation.

### Installation - Jupyter Notebook
This guide will help you set up the environment for the Dry Bean Classification project.
Python, Git and Jupyter Notebook are foundamental prerequisites.
Fist of all it is necessary to clone the Repository and move to the folder where it's saved:

```bash
git clone https://github.com/ChiccoSechi/dry-bean-ML-project.git
cd your-repository
```

and then, install the project dependecies listed in the `requirements.txt` file. Install them using pip:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- ucimlrepo
- numpy
- matplotlib
- pandas
- seaborn
- scikit-learn
- scipy
- imbalanced-learn
- notebook
- jupyterlab

Then open and run the project notebook (Jupyter Notebook):

```bash
jupyter notebook
```

This will start the Jupyter server and open a browser window. Navigate to the project notebook file (`.ipynb`) and open it.