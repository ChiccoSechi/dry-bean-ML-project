def conclusion(state):
    print("Home\Conclusion\n")
    
    print("Steps taken so far:")
    print("    Hyperparameters Tuning of the following algorithms:")
    print("        - Decision Tree,")
    print("        - K-Nearest Neighbors,")
    print("        - Naive Bayes (Gaussian),")
    print("        - Support Vector Classifier,")
    print("        - Random Forest,")
    print("        - Voting Classifier.")
    print("\n    With the following pre-processing technique combinations:")
    print("        - None,")
    print("        - standardization,")
    print("        - normalization,")
    print("        - undersampling,")
    print("        - oversampling,")
    print("        - standardization + undersampling,")
    print("        - standardization + oversampling,")
    print("        - normalization + undersampling,")
    print("        - normalization + oversampling,")
    print("        - standardization + agglomeration,")
    print("        - normalization + agglomeration,")
    print("        - standardization + agglomeration + undersampling,")
    print("        - standardization + agglomeration + oversampling,")
    print("        - normalization + agglomeration + undersampling,")
    print("        - normalization + agglomeration + oversampling.")
    print("\n    And various hyperparameters combinations:")
    print("        - Decision Tree:")
    print("            -> max_depth: depth of the classification tree;")
    print("            -> criterion: measure of split quality.")
    print("        - K-Nearest Neighbors: ")
    print("            -> n_neighbors: number of closest instances for classification;")
    print("            -> weights: weighting method for the closest instances;")
    print("            -> metric: distance function used to measure proximity between points in space.")
    print("        - Naive Bayes (Gaussian):")
    print("            -> var_smoothing: maximum variance of features added to variances to ensure stability.")
    print("        - Support Vector Classifier:")
    print("            -> kernel: function to map data by separating them linearly;")
    print("            -> C: classification error penalty;")
    print("            -> gamma: influence of each training point;")
    print("            -> class_weight: balances the importance of classes.")
    print("        - Random Forest:")
    print("            -> n_estimator: number of decision trees;")
    print("            -> criterion: measure of split quality;")
    print("            -> max_depth: maximum depth of trees;")
    print("            -> class_weight: balances the importance of classes.")
    print("        - Voting Classifier:")
    print("            -> voting: strategy for aggregating predictions;")
    print("            -> weights: balances the importance of classes.\n")
    

    best_list = sorted(state["best_list"], key=lambda x: x[1]["accuracy"].item(), reverse=True)
    
    for i, (model_name, model_info) in enumerate(best_list):
        accuracy = model_info["accuracy"].item()
        
        if i == 0:
            print(f"The best model is {model_name} with accuracy: {accuracy:.4f}")
        else:
            print(f"\nFollowed by {model_name} with accuracy: {accuracy:.4f}")
            
        if "technique" in model_info.columns:
            technique = model_info["technique"].item()
            print(f"Pre-processing techniques used: {technique}")

        print("Hyperparameters:")
        if model_name == "Decision Tree":
            print(f"   {'Max Depth:':<15} {model_info['max_depth'].item()}")
            print(f"   {'Criterion:':<15} {model_info['criterion'].item()}")

        elif model_name == "K-Nearest Neighbors":
            print(f"   {'Neighbors:':<15} {model_info['k_neighbors'].item()}")
            print(f"   {'Weights:':<15} {model_info['weights'].item()}")
            print(f"   {'Metric:':<15} {model_info['metric'].item()}")

        elif model_name == "Naive Bayes (Gaussian)":
            print(f"   {'Var Smoothing:':<15} {model_info['var_smoothing'].item()}")
            print(f"   {'Priors:':<15} {'Weights' if model_info['priors'].item() is not None else 'None'}")

        elif model_name == "Support Vector Classifier":
            print(f"   {'Kernel:':<15} {model_info['kernel'].item()}")
            print(f"   {'C:':<15} {model_info['C'].item()}")
            print(f"   {'Gamma:':<15} {model_info['gamma'].item()}")
            print(f"   {'Class Weight:':<15} {model_info['class_weight'].item()}")

        elif model_name == "Random Forest":
            print(f"   {'N Estimators:':<15} {model_info['n_estimators'].item()}")
            print(f"   {'Criterion:':<15} {model_info['criterion'].item()}")
            print(f"   {'Max Depth:':<15} {model_info['max_depth'].item()}")
            print(f"   {'Class Weight:':<15} {model_info['class_weight'].item()}")

        elif model_name == "Voting Classifier":
            print(f"   {'Voting Type:':<15} {model_info['voting'].item()}")
            print(f"   {'Weights:':<15} {model_info['weights'].item()}")