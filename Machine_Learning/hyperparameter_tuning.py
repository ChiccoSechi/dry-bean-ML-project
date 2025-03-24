from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from typing import Literal
from tqdm import tqdm
import pandas as pd
import numpy as np

techniques = [
    [],
    ["standardization"],
    ["normalization"],
    ["undersampling"],
    ["oversampling"],
    ["standardization", "undersampling"],
    ["standardization", "oversampling"],
    ["normalization", "undersampling"],
    ["normalization", "oversampling"],
    ["standardization", "agglomeration"],
    ["normalization", "agglomeration"],
    ["standardization", "agglomeration", "undersampling"],
    ["standardization", "agglomeration", "oversampling"],
    ["normalization", "agglomeration", "undersampling"],
    ["normalization", "agglomeration", "oversampling"]
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

def hyperparameter_tuning(X, y, classifier: Literal["DecisionTree", "K-NearestNeighbors", "NaiveBayes", "SupportVectorMachine", "RandomForest"]):
        
    if classifier in ["DecisionTree", "K-NearestNeighbors", "NaiveBayes"]:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
    if classifier == "SupportVectorMachine":
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.1, random_state=21, stratify=y)
        X_train, _, y_train, _ = train_test_split(X_subset, y_subset, test_size=0.2, random_state=21, stratify=y_subset)
    if classifier == "RandomForest":
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.3, random_state=21, stratify=y)
        X_train, _, y_train, _ = train_test_split(X_subset, y_subset, test_size=0.2, random_state=21, stratify=y_subset)
    
    if classifier == "DecisionTree":
        print("Home\Hyperparameter_Tuning\Decision_Tree\n")
        print("Tuning of the following hyperparameters via cross validation:")
        print("    max_depth: from 1 to 15.")
        print("    criterion: gini and entropy.\n")
        results_decisionTree = []
        
        clf = DecisionTreeClassifier(random_state=21)
        
        param_grid = {
        "classifier__max_depth": list(range(5,16)),
        "classifier__criterion": ['gini', 'entropy']
        }
        
        best_decisionTree_score = -1
        best_decisionTree_clf = None
        
    elif classifier == "K-NearestNeighbors":
        print("Home\Hyperparameter_Tuning\K_Nearest_Neighbors\n")
        print("Tuning of the following hyperparameters via cross validation:")
        print("    n_neighbors: from 1 to 20.")
        print("    weights: uniform and distance.")
        print("    metric: euclidean, manhattan and chebyshev.\n")
        results_knn = []
        
        clf = KNeighborsClassifier(n_jobs=-1)
        
        param_grid = {
            "classifier__n_neighbors": list(range(1, 21)),
            "classifier__weights": ['uniform', 'distance'],
            "classifier__metric": ['euclidean', 'manhattan', 'chebyshev']
        }
        
        best_knn_score = -1
        best_knn_clf = None
        
    elif classifier == "NaiveBayes":
        print("Home\Hyperparameter_Tuning\\Naive_Bayes\n")
        print("Tuning of the following hyperparameters via cross validation:")
        print("    var_smoothing: 1.e-17, 1.e-16, 1.e-15, 1.e-14, 1.e-13, 1.e-12, 1.e-11, 1.e-10 and 1.e-09.")
        print("    priors: weights inversely propotional to the classes and None.\n")
        results_naiveBayes = []
        
        clf = GaussianNB()
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )

        weights = np.round(class_weights / class_weights.sum(), 2)
        
        param_grid = {
            "classifier__var_smoothing": np.logspace(-17, -9, 9),
            "classifier__priors": [None, weights]
        }
        
        best_naiveBayes_score = -1
        best_naiveBayes_clf = None
        
    elif classifier == "SupportVectorMachine":
        print("Home\Hyperparameter_Tuning\Support_Vector_Machine\n")
        print("Tuning of the following hyperparameters via cross validation:")
        print("    kernel: linear or rbf.")
        print("    C: 0.1, 1 and 10.")
        print("    gamma: scale, auto, 0.1 and 1.")
        print("    class_weight: None and balanced.\n")
        results_svc = []
        
        clf = SVC(random_state=21)
        
        linear_params = {
            "classifier__kernel": ['linear'], 
            "classifier__C": np.logspace(-1, 1, 3), 
            "classifier__class_weight": [None, 'balanced']
        }
        rbf_params = {
            "classifier__kernel": ['rbf'], 
            "classifier__C": np.logspace(-1, 1, 3), 
            "classifier__gamma": ['scale', 'auto'] + list(np.logspace(-1, 0, 2)),
            "classifier__class_weight": [None, 'balanced']
            }

        param_grid = [linear_params, rbf_params]
        
        best_svc_score = -1
        best_svc_clf = None
        
    elif classifier == "RandomForest":
        print("Home\Hyperparameter_Tuning\Random_Forest\n")
        print("Tuning of the following hyperparameters via cross validation:")
        print("    n_estimators: 100, 200 and 300.")
        print("    criterion: gini and entropy.")
        print("    max_depth: 5, 10 and 15.")
        print("    class_weight: balanced and balanced_subsample.\n")
        results_randomForest = []
        
        clf = RandomForestClassifier(random_state=21, n_jobs=-1)
        
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__criterion": ['gini', 'entropy'],
            "classifier__max_depth": [5, 10, 15],
            "classifier__class_weight": ['balanced', 'balanced_subsample']
        }
        
        best_randomForest_score = -1
        best_randomForest_clf = None
        
    for technique in tqdm(techniques, bar_format="{l_bar}{bar:"+str(len(techniques))+"}{r_bar}"):
        print()
        steps = []

        if "standardization" in technique:
            steps.append(("scaler", StandardScaler()))
            n_clusters = 6

        elif "normalization" in technique:
            steps.append(("normalizer", MinMaxScaler()))
            n_clusters = 4

        if "agglomeration" in technique:
            steps.append(("agglomeration", FeatureAgglomeration(
                n_clusters=n_clusters,
                metric="euclidean",
                linkage="ward"
            )))

        if "undersampling" in technique:
            steps.append(("undersampler", ClusterCentroids(random_state=21)))

        elif "oversampling" in technique:
            steps.append(("oversampler", SMOTE(random_state=21)))

        steps.append(("classifier", clf))
        pipeline = ImbPipeline(steps)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring="balanced_accuracy",
            n_jobs=-1,                      
            verbose=1                       
        )
        
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        if classifier == "DecisionTree":
            if grid_search.best_score_ > best_decisionTree_score:
                best_decisionTree_score = grid_search.best_score_
                best_decisionTree_clf = grid_search.best_estimator_
                
            result = {
                "techniques_list": technique,                                   
                "techniques": ', '.join(technique),                             
                "max_depth": best_params["classifier__max_depth"],              
                "criterion": best_params["classifier__criterion"],              
                "accuracy": f"{grid_search.best_score_:.3f}",            
                "std": f"{grid_search.cv_results_['std_test_score'][grid_search.cv_results_['mean_test_score'].argmax()]:.3f}"         
            }
            
            results_decisionTree.append(result)
        
        elif classifier == "K-NearestNeighbors":
            if grid_search.best_score_ > best_knn_score:
                best_knn_score = grid_search.best_score_
                best_knn_clf = grid_search.best_estimator_
            
            result = {
                "techniques_list": technique,                                   
                "techniques": ', '.join(technique),                              
                "n_neighbors": best_params["classifier__n_neighbors"],           
                "weights": best_params["classifier__weights"],                  
                "metric": best_params["classifier__metric"],                    
                "accuracy": f"{grid_search.best_score_:.3f}",            
                "std": f"{grid_search.cv_results_['std_test_score'][grid_search.cv_results_['mean_test_score'].argmax()]:.3f}"  
            }
            
            results_knn.append(result)
            
        elif classifier == "NaiveBayes":
            if grid_search.best_score_ > best_naiveBayes_score:
                best_naiveBayes_score = grid_search.best_score_
                best_naiveBayes_clf = grid_search.best_estimator_
                
            result = {
                "techniques_list": technique,                                   
                "techniques": ', '.join(technique),                             
                "var_smoothing": best_params['classifier__var_smoothing'],      
                "priors": best_params['classifier__priors'],                   
                "accuracy": f"{grid_search.best_score_:.3f}",            
                "std": f"{grid_search.cv_results_['std_test_score'][grid_search.cv_results_['mean_test_score'].argmax()]:.3f}"
            }
            
            results_naiveBayes.append(result)
            
        elif classifier == "SupportVectorMachine":
            if grid_search.best_score_ > best_svc_score:
                best_svc_score = grid_search.best_score_
                best_svc_clf = grid_search.best_estimator_
                
            result = {
                "techniques_list": technique,                                        
                "techniques": ', '.join(technique),                                  
                "kernel": best_params["classifier__kernel"],                       
                "C": best_params["classifier__C"],                                   
                "gamma": best_params.get("classifier__gamma", "None"),               
                "class_weight": best_params["classifier__class_weight"],             
                "accuracy": f"{grid_search.best_score_:.3f}",            
                "std": f"{grid_search.cv_results_['std_test_score'][grid_search.cv_results_['mean_test_score'].argmax()]:.3f}"
            }
            
            results_svc.append(result)
                
        elif classifier == "RandomForest":
            if grid_search.best_score_ > best_randomForest_score:
                best_randomForest_score = grid_search.best_score_
                best_randomForest_clf = grid_search.best_estimator_

            result = {
                "techniques_list": technique,                           
                "techniques": ', '.join(technique),                     
                "n_estimators": best_params["classifier__n_estimators"],     
                "criterion": best_params["classifier__criterion"],           
                "max_depth": best_params["classifier__max_depth"],           
                "class_weight": best_params["classifier__class_weight"],     
                "accuracy": f"{grid_search.best_score_:.3f}",            
                "std": f"{grid_search.cv_results_['std_test_score'][grid_search.cv_results_['mean_test_score'].argmax()]:.3f}"
            }
            
            results_randomForest.append(result)
            
            
        del grid_search
    
    if classifier == "DecisionTree":
        results_decisionTree = pd.DataFrame(results_decisionTree)
        print("\nBest configurations of DecisionTree for each pre-processing techinques:")
        print(results_decisionTree[["techniques", "max_depth", "criterion", "accuracy", "std"]].sort_values("accuracy", ascending=False))
        
        return True, results_decisionTree, best_decisionTree_clf, best_decisionTree_score
    
    elif classifier == "K-NearestNeighbors":
        results_knn = pd.DataFrame(results_knn)
        print("\nBest configurations of K-NearestNeighbors for each pre-processing techinques:")
        print(results_knn[["techniques", "n_neighbors", "weights", "metric", "accuracy", "std"]].sort_values("accuracy", ascending=False)) 
        
        return True, results_knn, best_knn_clf, best_knn_score
    
    elif classifier == "NaiveBayes":
        results_naiveBayes = pd.DataFrame(results_naiveBayes)
        print("\nBest configurations of NaiveBayes for each pre-processing techinques:")
        print(results_naiveBayes[["techniques", "var_smoothing", "priors", "accuracy", "std"]].sort_values("accuracy", ascending=False)) 
        
        return True, results_naiveBayes, best_naiveBayes_clf, best_naiveBayes_score
    
    elif classifier == "SupportVectorMachine":
        results_svc = pd.DataFrame(results_svc)
        print("\nBest configurations of SupportVectorMachine for each pre-processing techinques:")
        print(results_svc[["techniques", "kernel", "C", "gamma", "class_weight", "accuracy", "std"]].sort_values("accuracy", ascending=False))

        return True, results_svc, best_svc_clf, best_svc_score
    
    elif classifier == "RandomForest":
        results_randomForest = pd.DataFrame(results_randomForest)
        print("\nBest configurations of RandomForest for each pre-processing techinques:")
        print(results_randomForest[["techniques", "n_estimators", "criterion", "max_depth", "class_weight", "accuracy", "std"]].sort_values("accuracy", ascending=False)) 
        
        return True, results_randomForest, best_randomForest_clf, best_randomForest_score
    
    
    
def voting_classifier(X, y, state):
    
    print("Home\Hyperparameter_Tuning\Voting_Classifier\n")
    print("Tuning of the following hyperparameters via cross validation:")
    print("    voting: hard and soft.")
    print("    weights: None, proportional to the results of the classifiers and more.\n")
    
    best_score_hard = [state['score_decisionTree'], state['score_knn'], state['score_naiveBayes'], state['score_svc'], state['score_randomForest']]
    best_score_soft = [state['score_decisionTree'], state['score_knn'], state['score_naiveBayes'], state['score_randomForest']]
    results_votingClassifier = []

    hard_voting_clf = VotingClassifier(
        estimators=[
            ("decisiontree", state["best_decisionTree_clf"]),
            ("knn", state["best_knn_clf"]),
            ("naivebayes", state["best_naiveBayes_clf"]),
            ("svc", state["best_svc_clf"]),
            ("randomforest", state["best_randomForest_clf"])
        ],
        voting='hard',
        n_jobs=-1
    )

    soft_voting_clf = VotingClassifier(
        estimators=[
            ("decisiontree", state["best_decisionTree_clf"]),
            ("knn", state["best_knn_clf"]),
            ("naivebayes", state["best_naiveBayes_clf"]),
            ("randomforest", state["best_randomForest_clf"])
        ],
        voting='soft',
        n_jobs=-1
    )

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

    hard_param_grid = {
        "weights": [
            None,
            best_score_hard,
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [2, 3, 1, 5, 4]
        ]
    }

    soft_param_grid = {
        "weights": [
            None,
            best_score_soft,
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [2, 3, 1, 4]
        ]
    }

    hard_grid_search = GridSearchCV(
        estimator=hard_voting_clf,
        param_grid=hard_param_grid,
        cv=skf,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )

    soft_grid_search = GridSearchCV(
        estimator=soft_voting_clf,
        param_grid=soft_param_grid,
        cv=skf,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )

    hard_grid_search.fit(X_train, y_train)
    soft_grid_search.fit(X_train, y_train)

    for params, mean_test_score, std_test_score in zip(
        hard_grid_search.cv_results_["params"],
        hard_grid_search.cv_results_["mean_test_score"],
        hard_grid_search.cv_results_["std_test_score"]
    ):
        result = {
            "voting": "hard",
            "weights": params["weights"],
            "accuracy": mean_test_score,
            "std": std_test_score
        }

        results_votingClassifier.append(result)

    del hard_grid_search

    for params, mean_test_score, std_test_score in zip(
        soft_grid_search.cv_results_["params"],
        soft_grid_search.cv_results_["mean_test_score"],
        soft_grid_search.cv_results_["std_test_score"]
    ):
        result = {
            "voting": "soft",                  
            "weights": params["weights"],            
            "accuracy": mean_test_score,     
            "std": std_test_score         
        }

        results_votingClassifier.append(result)

    del soft_grid_search
    
    results_votingClassifier = pd.DataFrame(results_votingClassifier)

    print("\nBest configurations of VotingClassifier:")
    print(results_votingClassifier.sort_values("accuracy", ascending=False))
    
    return True, results_votingClassifier 