from support_function import clear_terminal
from ucimlrepo import fetch_ucirepo 
from menu import main_menu_loop
from time import sleep
from tqdm import tqdm



if __name__ == "__main__":
    with tqdm(total=1, desc="Loading Dry bean dataset from UCIML... ") as pbar:
        dry_bean = fetch_ucirepo(id=602)
        pbar.update(1)
    
    v = dry_bean.variables
    X = dry_bean.data.features 
    y = dry_bean.data.targets 
    y = y.to_numpy().flatten()
    
    print("\n")
    print("Loaded.")
    sleep(2)
    clear_terminal()
    
    main_menu_loop(X, y, v)