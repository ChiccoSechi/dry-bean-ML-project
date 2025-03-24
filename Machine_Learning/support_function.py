import os
import send2trash

def clear_terminal():
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
def clear_directory():
    
    for dir in ["BOXPLOT", "HISTOGRAM", "CORRELATION MATRIX", "DENDROGRAM", "DECISION TREE", "K-NEAREST NEIGHBORS", "NAIVE BAYES", "SUPPORT VECTOR CLASSIFIER", "RANDOM FOREST", "VOTING CLASSIFIER"]:
        path_dir = os.path.join(os.getcwd(), dir)

        if os.path.exists(path_dir):
            send2trash.send2trash(path_dir)
            print(f"Directory '{dir}' sent to trash.")
        else:
            print(f"Directory '{dir}' does not exist.")