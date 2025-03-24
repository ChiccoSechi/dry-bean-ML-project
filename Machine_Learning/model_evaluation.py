import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import combine_preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from typing import Literal
from tqdm import tqdm
import pandas as pd
import os

def model_evaluation(X, y, state, classifier: Literal["decisionTree", "knn", "naiveBayes", "svc", "randomForest"]):
    
    if classifier == "decisionTree":
        print("Home\Model_Evaluation\Decision_Tree\n")
        dir = "DECISION TREE"
    elif classifier == "knn":
        print("Home\Model_Evaluation\K_Nearest_Neighbors\n")
        dir = "K-NEAREST NEIGHBORS"
    elif classifier == "naiveBayes":
        print("Home\Model_Evaluation\\Naive_Bayes\n")
        dir = "NAIVE BAYES"
    elif classifier == "svc":
        print("Home\Model_Evaluation\Support_Vector_Machine\n")
        dir = "SUPPORT VECTOR CLASSIFIER"
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.1, random_state=21, stratify=y)
    elif classifier == "randomForest":
        print("Home\Model_Evaluation\Random_Forest\n")
        dir = "RANDOM FOREST"
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.3, random_state=21, stratify=y)
    
    path_dir = os.path.join(os.getcwd(), dir)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    results = []
    
    
    for row in tqdm(state[f"results_{classifier}"].itertuples(), total=len(state[f"results_{classifier}"]), desc="Evaluation of the model...", bar_format="{l_bar}{bar:15}{r_bar}"):
        
        technique = row.techniques_list
        
        if classifier in ["decisionTree", "knn", "naiveBayes"]:
            X_train, X_test, y_train, y_test = combine_preprocessing(X, y, technique)
        elif classifier == "svc":
            X_train, X_test, y_train, y_test = combine_preprocessing(X_subset, y_subset, technique)
        elif classifier == "randomForest":
            X_train, X_test, y_train, y_test = combine_preprocessing(X_subset, y_subset, technique)    
        
        if classifier == "decisionTree":
            max_depth = row.max_depth
            criterion = row.criterion
            clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=21)
        elif classifier == "knn":
            k_neighbors = row.n_neighbors
            weights = row.weights
            metric = row.metric
            clf = KNeighborsClassifier(n_neighbors=k_neighbors, weights=weights, metric=metric, n_jobs=-1)
        elif classifier == "naiveBayes":
            var_smoothing = row.var_smoothing
            priors = row.priors
            clf = GaussianNB(var_smoothing=var_smoothing, priors=priors)
        elif classifier == "svc":
            c = row.C
            gamma = row.gamma
            class_weight = row.class_weight
            kernel = row.kernel
            if gamma == "None":
                clf = SVC(C=c, kernel=kernel, class_weight=class_weight, random_state=21)
            else:
                clf = SVC(C=c,kernel=kernel, gamma=gamma, class_weight=class_weight, random_state=21)
        elif classifier == "randomForest":
            n_estimators = row.n_estimators
            criterion = row.criterion
            max_depth = row.max_depth
            class_weight = row.class_weight
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, class_weight=class_weight, random_state=21, n_jobs=-1)
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        
        if classifier == "decisionTree":
            result = {
                "techniques_list": technique, 
                "technique": ', '.join(technique) if technique else 'no techniques',
                "accuracy": balanced_accuracy,
                "max_depth": max_depth,
                "criterion": criterion
            }    
        elif classifier == "knn":
            result = {
                "techniques_list": technique, 
                "technique": ', '.join(technique) if technique else 'no techniques',
                "accuracy": balanced_accuracy,
                "k_neighbors": k_neighbors,
                "weights": weights,
                "metric": metric
            }
        elif classifier == "naiveBayes":
            result = {
                "techniques_list": technique, 
                "technique": ', '.join(technique) if technique else 'no techniques',
                "accuracy": balanced_accuracy,
                "var_smoothing": var_smoothing,
                "priors": priors
            }
        elif classifier == "svc":
            result = {
                "techniques_list": technique, 
                "technique": ', '.join(technique) if technique else 'no techniques',
                "accuracy": balanced_accuracy,
                "kernel": kernel,
                "C": c,
                "gamma": gamma,
                "class_weight": class_weight
            }
        elif classifier == "randomForest":
            result = {
                "techniques_list": technique,
                "technique": ', '.join(technique) if technique else 'no techniques',
                "accuracy": balanced_accuracy,
                "n_estimators": n_estimators,
                "criterion": criterion,
                "max_depth": max_depth,
                "class_weight": class_weight
            }
        
        results.append(result)
        
        mcm = multilabel_confusion_matrix(y_test, y_pred)
        class_names = clf.classes_
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        if classifier == "decisionTree":
            fig.suptitle(f"Correlation Matrix Decision Trees\nTechniques: {', '.join(technique) if technique else 'no techniques'}\nMax_depth: {max_depth}\nCriterion: {criterion}\nAccuracy: {balanced_accuracy}")
        elif classifier == "knn":
            fig.suptitle(f"Correlation Matrix K-Nearest Neighbors\nTechniques: {', '.join(technique) if technique else 'no techniques'}\nK Neighbors: {k_neighbors}\nMetric: {metric} with {weights} weights\nAccuracy: {balanced_accuracy}")
        elif classifier == "naiveBayes":
            fig.suptitle(f"Correlation Matrix NaiveBayes\nTechniques: {', '.join(technique) if technique else 'no techniques'}\nVar Smoothing: {var_smoothing}\nPriors: {priors}\nAccuracy: {balanced_accuracy}")
        elif classifier == "svc":
            if gamma == "None":
                fig.suptitle(f"Correlation Matrix SVC\nTechniques: {', '.join(technique) if technique else 'no techniques'}\nKernel: {kernel}\nC: {c}\nClass weight: {class_weight}\nAccuracy: {balanced_accuracy}")
            else:
                fig.suptitle(f"Correlation Matrix SVC\nTechniques: {', '.join(technique) if technique else 'no techniques'}\nKernel: {kernel}\nC: {c}\nGamma: {gamma}\nClass weight: {class_weight}\nAccuracy: {balanced_accuracy}")
        elif classifier == "randomForest":
            fig.suptitle(f"Correlation Matrix RandomForest\nTechniques: {', '.join(technique) if technique else 'no techniques'}\nTrees: {n_estimators}\nCriterion: {criterion}\Profondità: {max_depth}\nClass weight: {class_weight}\nAccuracy: {balanced_accuracy}")
        
        axes = axes.flatten()
        
        for i, (matrix, class_name) in enumerate(zip(mcm, class_names)):
            tn, fp = matrix[0]
            fn, tp = matrix[1]
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            display_labels = [f"Not-{class_name}", f"{class_name}"]
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
            disp.plot(ax=axes[i], cmap="Reds")
            axes[i].set_title(f"{class_name} vs All\nAccuracy: {accuracy}")
        
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[7], cmap="Reds", values_format=".2f")
        axes[7].set_title(f"Confusion Matrix Normalized")
        axes[7].set_xticklabels(class_names, rotation=90)
        
        axes[8].set_visible(False)
        
        plt.figure(fig.number)
        plt.tight_layout()
        plt.savefig(f"{dir}/{'_'.join(technique) if technique else 'no_techniques'}_confusion_matrix")
        plt.close(fig)
    
    results = pd.DataFrame(results)
    print(f"\Evaluations of {dir.lower()} algorithm for the best combinations of hyperparameters:")
    if classifier == "decisionTree":
        print(results[["technique", "accuracy", "max_depth", "criterion"]].sort_values("accuracy", ascending=False))
    elif classifier == "knn":
        print(results[["technique", "accuracy", "k_neighbors", "weights", "metric"]].sort_values("accuracy", ascending=False))
    elif classifier == "naiveBayes":
        print(results[["technique", "accuracy", "var_smoothing", "priors"]].sort_values("accuracy", ascending=False))
    elif classifier == "svc":
        print(results[["technique", "accuracy", "kernel", "C", "gamma", "class_weight"]].sort_values("accuracy", ascending=False))
    elif classifier == "randomForest":
        print(results[["technique", "accuracy", "n_estimators", "criterion", "max_depth", "class_weight"]].sort_values("accuracy", ascending=False))
    
    print("\nBest Evaluation:")
    best_idx = results['accuracy'].idxmax()
    best_config = results.iloc[[best_idx]]
    if classifier == "decisionTree":
        print(best_config[["technique", "accuracy", "max_depth", "criterion"]])
    elif classifier == "knn":
        print(best_config[["technique", "accuracy", "k_neighbors", "weights", "metric"]])
    elif classifier == "naiveBayes":
        print(best_config[["technique", "accuracy", "var_smoothing", "priors"]])
    elif classifier == "svc":
        print(best_config[["technique", "accuracy", "kernel", "C", "gamma", "class_weight"]])
    elif classifier == "randomForest":
        print(best_config[["technique", "accuracy", "n_estimators", "criterion", "max_depth", "class_weight"]])
        
    return best_config

def voting_classifier_evaluation(X, y, state):
    
    print("Home\Model_Evaluation\Voting_Classifier\n")
    
    dir = "VOTING CLASSIFIER"
    
    path_dir = os.path.join(os.getcwd(), dir)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        
    results = []
    
    for i, row in tqdm(enumerate(state["results_votingClassifier"].itertuples()), total=len(state[f"results_votingClassifier"]), desc="Evaluation of the model...", bar_format="{l_bar}{bar:15}{r_bar}"):
        voting = row.voting
        weights = row.weights
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

        if voting == "soft":
            clf = VotingClassifier(
                estimators=[
                    ("decisiontree", state["best_decisionTree_clf"]),
                    ("knn", state["best_knn_clf"]),
                    ("naivebayes", state["best_naiveBayes_clf"]),
                    ("randomforest", state["best_randomForest_clf"])
                ],
                voting=voting,
                weights=weights,
                n_jobs=-1
            )
        elif voting == "hard":
            clf = VotingClassifier(
                estimators=[
                    ("decisiontree", state["best_decisionTree_clf"]),
                    ("knn", state["best_knn_clf"]),
                    ("naivebayes", state["best_naiveBayes_clf"]),
                    ("svc", state["best_svc_clf"]),
                    ("randomforest", state["best_randomForest_clf"])
                ],
                voting=voting,
                weights=weights,
                n_jobs=-1
            )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        
        result = {
            "accuracy": balanced_accuracy,
            "voting": voting,
            "weights": weights
        }
        
        results.append(result)
        mcm = multilabel_confusion_matrix(y_test, y_pred)
        class_names = clf.classes_

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f"Correlation Matrix VotingClassifier\nVoting: {voting}\nWeights: {weights}")
        axes = axes.flatten()
        
        for j, (matrix, class_name) in enumerate(zip(mcm, class_names)):
            tn, fp = matrix[0]
            fn, tp = matrix[1]
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            display_labels = [f"Not-{class_name}", f"{class_name}"]
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
            disp.plot(ax=axes[j], cmap="Reds")
            axes[j].set_title(f"{class_name} vs All\nAccuracy: {accuracy}")

        cm = confusion_matrix(y_test, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[7], cmap="Reds", values_format=".2f")
        axes[7].set_title(f"Confusion Matrix Normalized")
        axes[7].set_xticklabels(class_names, rotation=90)
        axes[8].set_visible(False)
        
        plt.figure(fig.number)
        plt.tight_layout()
        plt.savefig(f"{dir}/{i}_confusion_matrix")
        plt.close(fig)
        
    results = pd.DataFrame(results)
    print(f"\Evaluations of voting classifier algorithm for the best combinations of hyperparameters:")
    print(results.sort_values("accuracy", ascending=False))
    
    print("\nBest Evaluation:")
    best_idx = results['accuracy'].idxmax()
    best_config = results.iloc[[best_idx]]
    best_config = results.iloc[[best_idx]]
    print(best_config)
     
    return best_config
    