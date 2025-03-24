import matplotlib
matplotlib.use('Agg')
from preprocessing import standardization, normalization
from support_function import clear_terminal
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import os

def beans_counts_function(y):
    
    beans, counts = np.unique(y, return_counts=True)
    sort = np.argsort(counts)
    beans = beans[sort]
    counts = counts[sort]
    
    return beans, counts
    
def data_analysis(X, y, v):
    
    print("Home\Data_Analysis\Dataset_Info\n")
    print(f"Samples:{X.shape[0]}.")
    print(f"Features: {X.shape[1]}.")
    print("\nFeatures description:")
    
    for i in range(len(v)):
        print(f"{v['name'][i]:<15} -> {v['description'][i]}.")
        
    print("\nNumber of occurrences by class:")
    beans, counts = beans_counts_function(y)
    
    for bean, count in zip(beans, counts):
       print(f"{bean:<10} -> {count} occurrences.")
    
def boxplot_and_outliers(X, y):
    
    print("Home\Data_Analysis\Boxplot_Outliers\n")
    dir = "BOXPLOT"
    path_dir = os.path.join(os.getcwd(), dir)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    else:
        return print("The boxplots are alredy viewable in the \"BOXPLOT\" folder!")

    beans, counts = beans_counts_function(y)
    outliers = np.zeros((len(X.columns), len(beans) + 1), dtype=int)
    features_group1 = X.columns[:8]  
    features_group2 = X.columns[8:]  

    for j in tqdm(range(len(beans) + 1), desc="Generating boxplots by class and counting the outliers...", bar_format="{l_bar}{bar:"+str(len(beans)+1)+"}{r_bar}"):
        fig1, axs1 = plt.subplots(2, 4, figsize=(20, 10))

        if j == len(beans):
            fig1.suptitle(f"Boxplot - General 1", fontsize=25)
        else:
            fig1.suptitle(f"Boxplot - Class: {beans[j]} 1", fontsize=25)
        axs1 = axs1.flatten()  

        fig2, axs2 = plt.subplots(2, 4, figsize=(20, 10))

        if j == len(beans):
            fig2.suptitle(f"Boxplot - General 2", fontsize=25)
        else:
            fig2.suptitle(f"Boxplot - Class: {beans[j]} 2", fontsize=25)
        axs2 = axs2.flatten()  
        
        for i, feature in enumerate(features_group1):
            ax = axs1[i]  

            if j == len(beans):
                datas = X[feature]
            else:
                datas = X.iloc[(y == beans[j]), X.columns.get_loc(feature)]
            
                ax.boxplot(datas, 
                       showmeans=True,  
                       meanline=True,   
                       flierprops=dict(marker="x", markeredgecolor="purple"),
                       meanprops=dict(color="red", linestyle="--"),            
                       medianprops=dict(color="blue", linestyle="--"))         

            ax.set_title(f"Feature: {feature}") 

            Q1 = datas.quantile(0.25)  
            Q3 = datas.quantile(0.75)  
            IQR = Q3 - Q1              

            low_outliers = datas[datas < Q1 - 1.5 * IQR]           
            high_outliers = datas[datas > Q3 + 1.5 * IQR]          
            tot_outliers = len(low_outliers) + len(high_outliers)  

            feature_idx = X.columns.get_loc(feature)
            
            if j == len(beans):
                outliers[feature_idx][-1] = tot_outliers
            else:
                outliers[feature_idx][j] = tot_outliers

        for i, feature in enumerate(features_group2):
            ax = axs2[i]

            if j == len(beans):
                datas = X[feature]
            else:
                datas = X.iloc[(y == beans[j]), X.columns.get_loc(feature)]

            ax.boxplot(datas, 
                       showmeans=True, 
                       meanline=True,
                      flierprops=dict(marker="x", markeredgecolor="purple"),
                      meanprops=dict(color="red", linestyle="--"),
                      medianprops=dict(color="blue", linestyle="--")) 

            ax.set_title(f"Feature: {feature}")

            Q1 = datas.quantile(0.25)
            Q3 = datas.quantile(0.75)
            IQR = Q3 - Q1

            low_outliers = datas[datas < Q1 - 1.5 * IQR]
            high_outliers = datas[datas > Q3 + 1.5 * IQR]
            tot_outliers = len(low_outliers) + len(high_outliers)

            feature_idx = X.columns.get_loc(feature)
            
            if j == len(beans):
                outliers[feature_idx][-1] = tot_outliers
            else:
                outliers[feature_idx][j] = tot_outliers
    
        handles1 = [
            plt.Line2D([0], [0], color="red", linestyle="--", label="Media"),
            plt.Line2D([0], [0], color="blue", linestyle="--", label="Mediana"),
            plt.Line2D([0], [0], marker="x", color="purple", linestyle="", label="Outliers (x)", markersize=10)
        ]
        fig1.legend(handles=handles1, loc='upper right', ncol=3, fontsize=15)

        handles2 = [
            plt.Line2D([0], [0], color="red", linestyle="--", label="Media"),
            plt.Line2D([0], [0], color="blue", linestyle="--", label="Mediana"),
            plt.Line2D([0], [0], marker="x", color="purple", linestyle="", label="Outliers (x)", markersize=10)
        ]
        fig2.legend(handles=handles2, loc="upper right", ncol=3, fontsize=15)

        plt.figure(fig1.number)
        
        if j == len(beans):
            plt.savefig(f"{dir}/ALL_features_1_8")
        else:
            plt.savefig(f"{dir}/{beans[j]}_features_1_8")

        plt.figure(fig2.number)
        
        if j == len(beans):
            plt.savefig(f"{dir}/ALL_features_9_16")
        else:
            plt.savefig(f"{dir}/{beans[j]}_features_9_16")

        plt.close(fig1)
        plt.close(fig2)
        
    print("\nAll the Boxplots are saved in the \"BOXPLOT\" folder.\n")
    sleep(1)
    print("Outliers for each class and feature:")

    columns_names = list(beans) + ["General"]
    outliers_df = pd.DataFrame(
        data=outliers, 
        index=X.columns,       
        columns=columns_names
    )

    outliers_df["TOT (without General)"] = outliers_df.iloc[:, :-1].sum(axis=1)  

    print(outliers_df.to_string())
    sleep(1)
    print("\nCorrisponding percentage:")
    beans_counts = dict(zip(beans, counts))

    beans_counts_with_general = beans_counts.copy()
    beans_counts_with_general["General"] = sum(counts)

    outliers_df_perc = outliers_df.iloc[:, :-1].div(pd.Series(beans_counts_with_general), axis=1) * 100

    totale_campioni = sum(counts)
    outliers_df_perc["TOT (without General)"] = (outliers_df["TOT (without General)"] / totale_campioni * 100).round(2)

    outliers_df_perc = outliers_df_perc.round(2)

    print(outliers_df_perc.to_string())

def histogram(X, y):
    
    print("Home\Data_Analysis\Histogram\n")
    
    dir = "HISTOGRAM"
    path_dir = os.path.join(os.getcwd(), dir)

    if not os.path.exists(path_dir):
       os.makedirs(path_dir)
    else:
        return print("The histograms are alredy viewable in the \"HISTOGRAM\" folder!")
    
    beans, _ = beans_counts_function(y)
    
    features_group1 = X.columns[:8]  
    features_group2 = X.columns[8:]  

    for j in tqdm(range(len(beans) + 1), desc="Generating histograms by class...", bar_format="{l_bar}{bar:"+str(len(beans)+1)+"}{r_bar}"):
       fig1, axs1 = plt.subplots(2, 4, figsize=(20, 10))
    
       if j == len(beans):
          fig1.suptitle(f"Histogram - General 1", fontsize=25)
       else:
          fig1.suptitle(f"Histogram - Class: {beans[j]} 1", fontsize=25)
       
       axs1 = axs1.flatten() 
       fig2, axs2 = plt.subplots(2, 4, figsize=(20, 10))
    
       if j == len(beans):
          fig2.suptitle(f"Histogram - General 2", fontsize=25)
       else:
          fig2.suptitle(f"Histogram - Class: {beans[j]} 2", fontsize=25)
       
       axs2 = axs2.flatten()  
       
       for i, feature in enumerate(features_group1):
          ax = axs1[i] 

          if j == len(beans):
             datas = X[feature]
          else:
             datas = X.iloc[(y == beans[j]), X.columns.get_loc(feature)]

          mean = np.mean(datas)
          std = np.std(datas)

          ax.hist(datas, bins="auto", edgecolor="black", color="purple", alpha=0.5)
          ax.axvline(mean, color="red", linestyle="--")  
          ax.axvline(mean - std, color="blue", linestyle="dashed")  
          ax.axvline(mean + std, color="blue", linestyle="dashed")  

          if j == len(beans):
             ax.set_title(f"Feature: {feature}\nμ = {mean:.2f} - σ = {std:.2f}") 
          else:
             ax.set_title(f"Feature: {feature}\nClass {beans[j]}: μ = {mean:.2f} - σ = {std:.2f}") 

       for i, feature in enumerate(features_group2):
          ax = axs2[i] 

          if j == len(beans):
             datas = X[feature]
          else:
             datas = X.iloc[(y == beans[j]), X.columns.get_loc(feature)]

          mean = np.mean(datas)
          std = np.std(datas)

          ax.hist(datas, bins="auto", edgecolor="black", color="purple", alpha=0.5)
          ax.axvline(mean, color="red", linestyle="--")  
          ax.axvline(mean - std, color="blue", linestyle="dashed")  
          ax.axvline(mean + std, color="blue", linestyle="dashed")  

          if j == len(beans):
             ax.set_title(f"Feature: {feature}\nμ = {mean:.2f} - σ = {std:.2f}") 
          else:
             ax.set_title(f"Feature: {feature}\nClass {beans[j]}: μ = {mean:.2f} - σ = {std:.2f}") 
    
       handles1 = [
          plt.Line2D([0], [0], color="red", linestyle="--", label="Media (μ)"),
          plt.Line2D([0], [0], color="blue", linestyle="--", label="Deviazione Standard (σ)"),
       ]
       fig1.legend(handles=handles1, loc='upper right', ncol=2, fontsize=15)

       handles2 = [
          plt.Line2D([0], [0], color="red", linestyle="--", label="Media (μ)"),
          plt.Line2D([0], [0], color="blue", linestyle="--", label="Deviazione Standard (σ)"),
       ]
       fig2.legend(handles=handles2, loc='upper right', ncol=2, fontsize=15)
    
       plt.figure(fig1.number)
       if j == len(beans):
          plt.savefig(f"{dir}/ALL_features_1_8")
       else:
          plt.savefig(f"{dir}/{beans[j]}_features_1_8")
    
       plt.figure(fig2.number)
       if j == len(beans):
          plt.savefig(f"{dir}/ALL_features_9_16")
       else:
          plt.savefig(f"{dir}/{beans[j]}_features_9_16")
    
       plt.close(fig1)
       plt.close(fig2)
    
    print("\nAll the Histograms are saved in the \"HISTOGRAM\" folder.\n")
    
def correlation_matrix(X, y):
    
    print("Home\Data_Analysis\Correlation_Matrix\n")

    dir = "CORRELATION MATRIX"
    path_dir = os.path.join(os.getcwd(), dir)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    else:
        return print("The correletion matrices are alredy viewable in the \"CORRELATION MATRIX\" folder!")

    beans, _ = beans_counts_function(y)
    
    for j in tqdm(range(len(beans)), desc="Generating correlation matrix by class...", bar_format="{l_bar}{bar:"+str(len(beans))+"}{r_bar}"):
        datas = X.iloc[(y == beans[j])]
        correlation_matrix = datas.corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(f"Feature Correlation Matrix - Class {beans[j]}", fontsize=25)
        plt.savefig(f"{dir}/correlation_matrix_{beans[j]}.png")
        plt.close()

    print("Generating the general correlation matrix...")
    correlation_matrix_all = X.corr()

    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix_all, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("General Correlation Matrix", fontsize=25)
    plt.savefig(f"{dir}/ALL_correlation_matrix.png")
    plt.close()
    sleep(1)
    print("\nAll the Correlation Matrices are saved in the \"CORRELATION MATRIX\" folder.\n")
    
def dendrogram(X):
    
    print("Home\Data_Analysis\Dendrogram\n")
    
    dir = "DENDROGRAM"
    path_dir = os.path.join(os.getcwd(), dir)

    n_clusters_std = 6
    n_clusters_norm = 4
    
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    else:
        print("The dendrograms are alredy viewable in the \"DENDROGRAM\" folder!")
        print("Do you want to view a new dendrogram with different clusters?(y/n)")
        choice = input()
        
        if choice in ["y", "Y", "yes"]:
            clear_terminal()
            print("Home\Data_Analysis\Dendrogram\n")
            print("Note that this is for visualization puroses only,\nthe optimal number of clusters will remain unchanged.\nStandardization clusters = 6\nNormalization clusters = 4")
            n_clusters_std = input("Dendrogram clusters for standardization:")
            
            if n_clusters_std.isdigit():
                n_clusters_std = int(n_clusters_std)
            else:
                clear_terminal()
                print("Home\Data_Analysis\Dendrogram\n")
                return print("The value must be an integer!")
            
            n_clusters_norm = input("Dendrogram clusters for standardization:")
            
            if n_clusters_norm.isdigit():
                n_clusters_norm = int(n_clusters_norm)
            else:
                return print("The value must be an integer!")
        
        elif choice in ["n", "N", "no"]:
            return print("The dendrograms are alredy viewable in the \"DENDROGRAM\" folder!")
        else:
            return print("INVALID INPUT.")
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    distance_matrix_std = hierarchy.distance.pdist(standardization(X).T)
    linkage_matrix_std = hierarchy.linkage(distance_matrix_std, method="ward")

    hierarchy.dendrogram(
        linkage_matrix_std,
        labels=X.columns,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0,
        ax=axs[0] 
    )

    
    axs[0].set_title("Dendrogram (StandardScaler)", fontsize=15)
    axs[0].set_xlabel("Feature")
    axs[0].set_ylabel("Dissimilarity")

    distance_matrix_norm = hierarchy.distance.pdist(normalization(X).T)
    linkage_matrix_norm = hierarchy.linkage(distance_matrix_norm, method="ward")

    hierarchy.dendrogram(
        linkage_matrix_norm,
        labels=X.columns,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0,
        ax=axs[1]  
    )

    axs[1].set_title("Dendrogram (MinMaxScaler)", fontsize=15)
    axs[1].set_xlabel("Feature")
    axs[1].set_ylabel("Dissimilariy")

    fig.suptitle("Dendrogram with standardization - Dendrogram with normalization", fontsize=20)

    threshold_std = linkage_matrix_std[-(n_clusters_std-1), 2]

    hierarchy.dendrogram(
        linkage_matrix_std,
        labels=X.columns,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=threshold_std,
        ax=axs[2]  
    )

    axs[2].set_title("Dendrogram (StandardScaler)", fontsize=15)
    axs[2].set_xlabel("Feature")
    axs[2].set_ylabel("Dissimilarity")

    threshold_norm = linkage_matrix_norm[-(n_clusters_norm-1), 2]

    hierarchy.dendrogram(
        linkage_matrix_norm,
        labels=X.columns,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=threshold_norm,
        ax=axs[3]  
    )

    axs[3].set_title("Dendrogram (MinMaxScaler)", fontsize=15)
    axs[3].set_xlabel("Feature")
    axs[3].set_ylabel("Dissimilariy")
    fig.suptitle("Dendrogram with standardization - Dendrogram with normalization", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{dir}/Dendrogram_colored.png")
    plt.close()
    
    print("Cluster obtained from aggregated datas with standardization:")

    cluster_labels = hierarchy.fcluster(linkage_matrix_std, n_clusters_std, criterion='maxclust')

    unique_clusters = np.unique(cluster_labels)

    for i in unique_clusters:
       cluster_members = [col for col, label in zip(X.columns, cluster_labels) if label == i]
       print(f"Cluster {i}: {cluster_members}")

    print("\nCluster obtained from aggregated datas with normalization:")

    cluster_labels = hierarchy.fcluster(linkage_matrix_norm, n_clusters_norm, criterion='maxclust')
    unique_clusters = np.unique(cluster_labels)

    for i in unique_clusters:
       cluster_members = [col for col, label in zip(X.columns, cluster_labels) if label == i]
       print(f"Cluster {i}: {cluster_members}")
       
    print("\nThe Dendrograms are saved in the \"DENDROGRAM\" folder.\n")