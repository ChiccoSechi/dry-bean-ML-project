from support_function import clear_terminal, clear_directory
from hyperparameter_tuning import *
from conclusion import conclusion
from model_evaluation import *
from data_analysis import *

def start_menu():
    while True:
        print("Home\n")
        print("Dry bean dataset Machine Learning project.\n")
        print("Choose one of the options:")
        print("    1. DATA ANALYSIS")
        print("    2. HYPERPARAMETER TUNING")
        print("    3. MODEL EVALUATION")
        print("    4. CONCLUSION")
        print("    Q. EXIT")
        
        choice = input("Option:")
        
        if choice in ["1", "2", "3", "4", "Q", "q"]:
            return choice
        else:
            clear_terminal()
            print("INVALID INPUT.\nPlease enter a valid input.")
            print("\nPress Enter to go to -> Home...")
            input()
            clear_terminal()

def data_analysis_menu():
    while True:
        print("Home\Data_Analysis\n")
        print("Choose one of the options:")
        print("    1. DATASET INFO")
        print("    2. BOXPLOT")
        print("    3. HISTOGRAM")
        print("    4. CORRELATION MATRIX")
        print("    5. AGGLOMERATION")
        print("    Q. MAIN MENU")
        
        choice = input("Option:")
        
        if choice in ["1", "2", "3", "4", "5", "Q", "q"]:
            return choice
        else:
            clear_terminal()
            print("INVALID INPUT.\nPlease enter a valid input.")
            print("\nPress Enter to go to -> Home\Data_Analysis...")
            input()
            clear_terminal()
            
            
def hyperpameter_tuning_menu():
    while True:
        print("Home\Hyperparameter_Tuning\n")
        print("Choose one of the options:")
        print("    1. Decision Tree")
        print("    2. K-Nearest Neighbors")
        print("    3. Naive Bayes")
        print("    4. Support Vector Machine")
        print("    5. Random Forest")
        print("    6. Voting Classifier")
        print("    Q. MAIN MENU")
        
        choice = input("Option:")
        
        if choice in ["1", "2", "3", "4", "5", "6", "Q", "q"]:
            return choice
        else:
            clear_terminal()
            print("INVALID INPUT.\nPlease enter a valid input.")
            print("\nPress Enter to go to -> Home\Hyperparameter_Tuning...")
            input()
            clear_terminal()
            
def model_evaluation_menu():
    while True:
        print("Home\Model_Evaluation\n")
        print("Choose one of the options:")
        print("    1. Decision Tree")
        print("    2. K-Nearest Neighbors")
        print("    3. Naive Bayes")
        print("    4. Support Vector Machine")
        print("    5. Random Forest")
        print("    6. Voting Classifier")
        print("    Q. MAIN MENU")
        
        choice = input("Option:")
        
        if choice in ["1", "2", "3", "4", "5", "6", "Q", "q"]:
            return choice
        else:
            clear_terminal()
            print("INVALID INPUT.\nPlease enter a valid input.")
            print("\nPress Enter to go to -> Home\Model_Evaluation...")
            input()
            clear_terminal()
            
def main_menu_loop(X, y, v):
    
    state = {
        "bool_decisionTree": False,
        "results_decisionTree": None,   
        "best_decisionTree_clf": None, 
        "score_decisionTree": None,
            
        "bool_knn": False,
        "results_knn": None, 
        "best_knn_clf": None,
        "score_knn": None,
        
        "bool_naiveBayes": False,
        "results_naiveBayes": None,
        "best_naiveBayes_clf": None,
        "score_naiveBayes": None,
        
        "bool_svc": False,
        "results_svc": None,
        "best_svc_clf": None,
        "score_svc": None,
        
        "bool_randomForest": False,
        "results_randomForest": None,
        "best_randomForest_clf": None,
        "score_randomForest": None,
        
        "bool_votingClassifier": False,
        "results_votingClassifier": None,
        
        "best_list": []
    }
    
    while True:
        clear_terminal()
        choice = start_menu()
        
        if choice == "1":
            data_analysis_menu_loop(X, y, v)
            
        elif choice == "2":
            hyperparameter_tuning_menu_loop(X, y, state)
            
        elif choice == "3":
            model_evaluation_menu_loop(X, y, state)
            
        elif choice == "4":
            conclusion(state)
            print("\nPress Enter to go to -> Home...")
            input()
            
        elif choice.upper() == "Q":
            print("Do you want to delete all the folders?(y/n)")
            choice2 = input()
            
            if choice2 in ["y", "Y", "yes"]:
                clear_directory()
                print("Thanks for using the program!\nClosing the program...")
                break
            elif choice2 in ["n", "N", "no"]:
                print("Thanks for using the program!\nClosing the program...")
                break
            else:
                clear_terminal()
                print("INVALID INPUT.\nPlease enter a valid input.")
                print("\nPress Enter to go to -> Home...")
                input()
                clear_terminal()
            

def data_analysis_menu_loop(X, y, v):
    while True:
        clear_terminal()
        choice = data_analysis_menu()
        
        if choice == "1":
            clear_terminal()
            data_analysis(X, y, v)
            print("\nPress Enter to go to -> Home\Data_Analysis...")
            input()
            
        elif choice == "2":
            clear_terminal()
            boxplot_and_outliers(X, y)
            print("\nPress Enter to go to -> Home\Data_Analysis...")
            input()
            
        elif choice == "3":
            clear_terminal()
            histogram(X, y)
            print("\nPress Enter to go to -> Home\Data_Analysis...")
            input()
        
        elif choice == "4":
            clear_terminal()
            correlation_matrix(X, y)
            print("\nPress Enter to go to -> Home\Data_Analysis...")
            input()
        
        elif choice == "5":
            clear_terminal()
            dendrogram(X)
            print("\nPress Enter to go to -> Home\Data_Analysis...")
            input()
            
        elif choice.upper() == "Q":
            break

def hyperparameter_tuning_menu_loop(X, y, state):
    while True:
        clear_terminal()
        choice = hyperpameter_tuning_menu()
        
        if choice == "1":
            clear_terminal()
            if state["bool_decisionTree"] == True:
                print("Home\Hyperparameter_Tuning\Decision_Tree\n")
                print("Decision Tree hyperparameter tuning already done.")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
            elif state["bool_decisionTree"] == False:
                state["bool_decisionTree"], state["results_decisionTree"], state["best_decisionTree_clf"], state["score_decisionTree"]= hyperparameter_tuning(X, y, "DecisionTree")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
            
        elif choice == "2":
            clear_terminal()
            if state["bool_knn"] == True:
                print("Home\Hyperparameter_Tuning\K-K_Nearest_Neighbors\n")
                print("K-Nearest Neighbors hyperparameter tuning already done.")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
            elif state["bool_knn"] == False:
                state["bool_knn"], state["results_knn"], state["best_knn_clf"], state["score_knn"] = hyperparameter_tuning(X, y, "K-NearestNeighbors")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
            
        elif choice == "3":
            clear_terminal()
            if state["bool_naiveBayes"] == True:
                print("Home\Hyperparameter_Tuning\\Naive_Bayes\n")
                print("Naive Bayes hyperparameter tuning already done.")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
            elif state["bool_naiveBayes"] == False:
                state["bool_naiveBayes"], state["results_naiveBayes"], state["best_naiveBayes_clf"], state["score_naiveBayes"] = hyperparameter_tuning(X, y, "NaiveBayes")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
        
        elif choice == "4":
            clear_terminal()
            if state["bool_svc"] == True:
                print("Home\Hyperparameter_Tuning\Support_Vector_Machine\n")
                print("Support Vector Machine hyperparameter tuning already done.")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
            elif state["bool_svc"] == False:
                state["bool_svc"], state["results_svc"], state["best_svc_clf"], state["score_svc"] = hyperparameter_tuning(X, y, "SupportVectorMachine")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
        
        elif choice == "5":
            clear_terminal()
            if state["bool_randomForest"] == True:
                print("Home\Hyperparameter_Tuning\Random_Forest\n")
                print("Random Forest hyperparameter tuning already done.")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
            elif state["bool_randomForest"] == False:
                state["bool_randomForest"], state["results_randomForest"], state["best_randomForest_clf"], state["score_randomForest"] = hyperparameter_tuning(X, y, "RandomForest")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
                
        elif choice == "6":
            clear_terminal()
            if all(state[f"bool_{model}"] for model in ["decisionTree", "knn", "naiveBayes", "svc", "randomForest"]):
                if state["bool_votingClassifier"] == True:
                    print("Home\Hyperparameter_Tuning\Voting_Classifier\n")
                    print("Voting Classifier hyperparameter tuning already done.")
                    print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                    input()
                elif state["bool_votingClassifier"] == False:
                    state["bool_votingClassifier"], state["results_votingClassifier"] = voting_classifier(X, y, state)
                    print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                    input()
            else:
                print("Home\Hyperparameter_Tuning\Voting_Classifier\n")
                print("The hyperparameters of the other algorithms need to be tuned first.")
                for model in ["decisionTree", "knn", "naiveBayes", "svc", "randomForest"]:
                    if state[f"bool_{model}"]:
                        print(f"{model}: Done.")
                    else:
                        print(f"{model}: Not done.")
                print("\nPress Enter to go to -> Home\Hyperarameter_Tuning...")
                input()
                
        elif choice.upper() == "Q":
            break

def model_evaluation_menu_loop(X, y, state):
    while True:
        clear_terminal()
        choice = model_evaluation_menu()
        
        if choice == "1":
            clear_terminal()
            if state["bool_decisionTree"] == True:
                best_decisionTree = model_evaluation(X, y, state, "decisionTree")
                state["best_list"].append(("Decision Tree", best_decisionTree))
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            elif state["bool_decisionTree"] == False:
                print("Home\Model_Evaluation\Decision_Tree\n")
                print(" Decision Tree hyperparameter tuning not done.")
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
                
        elif choice == "2":
            clear_terminal()
            if state["bool_knn"] == True:
                best_knn = model_evaluation(X, y, state, "knn")
                state["best_list"].append(("K-Nearest Neighbors", best_knn))
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            elif state["bool_knn"] == False:
                print("Home\Model_Evaluation\K_Nearest_Neighbors\n")
                print("K-Nearest Neighbors hyperparameter tuning not done.")
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            
        elif choice == "3":
            clear_terminal()
            if state["bool_naiveBayes"] == True:
                best_naiveBayes = model_evaluation(X, y, state, "naiveBayes")
                state["best_list"].append(("Naive Bayes (Gaussian)", best_naiveBayes))
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            elif state["bool_naiveBayes"] == False:
                print("Home\Model_Evaluation\\Naive_Bayes\n")
                print("Naive Bayes hyperparameter tuning not done.")
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
        
        elif choice == "4":
            clear_terminal()
            if state["bool_svc"] == True:
                best_svc = model_evaluation(X, y, state, "svc")
                state["best_list"].append(("Support Vector Classifier", best_svc))
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            elif state["bool_svc"] == False:
                print("Home\Model_Evaluation\Support_Vector_Machine\n")
                print("Support Vector Machine hyperparameter tuning not done.")
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
        
        elif choice == "5":
            clear_terminal()
            if state["bool_randomForest"] == True:
                best_randomForest = model_evaluation(X, y, state, "randomForest")
                state["best_list"].append(("Random Forest", best_randomForest))
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            elif state["bool_randomForest"] == False:
                print("Home\Model_Evaluation\Random_Forest\n")
                print("Random Forest hyperparameter tuning not done.")
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
                
        elif choice == "6":
            clear_terminal()
            if all(state[f"bool_{model}"] for model in ["decisionTree", "knn", "naiveBayes", "svc", "randomForest"]) and state["bool_votingClassifier"]:
                best_voting = voting_classifier_evaluation(X, y, state)
                state["best_list"].append(("Voting Classifier", best_voting))
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            else:
                print("Home\Model_Evaluation\Random_Forest\n")
                print("Hyperparameter tuning must first be performed for all algorithms!")
                for model in ["decisionTree", "knn", "naiveBayes", "svc", "randomForest", "votingClassifier"]:
                    if state[f"bool_{model}"]:
                        print(f"{model}: Done.")
                    else:
                        print(f"{model}: Not done.")
                print("\nPress Enter to go to -> Home\Model_Evaluation...")
                input()
            
        elif choice.upper() == "Q":
            break