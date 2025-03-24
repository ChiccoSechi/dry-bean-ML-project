from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import train_test_split
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from typing import Literal


def standardization(X_train, X_test=None):
    
    scaler = StandardScaler()   

    X_train_std = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_std = scaler.transform(X_test)
        return X_train_std, X_test_std
    else:
        return X_train_std

def normalization(X_train, X_test=None):
    
    normalizer = MinMaxScaler()
    
    X_train_norm = normalizer.fit_transform(X_train)
    
    if X_test is not None:
        X_test_norm = normalizer.transform(X_test)
        return X_train_norm, X_test_norm
    else:
        return X_train_norm

def agglomeration(X_train, X_test, type: Literal["normalization", "standardization"]):
    
    if type not in ["normalization", "standardization"]:
        raise ValueError("cluster_type deve essere 'normalization' o 'standardization'")
    
    if type == "normalization":
        clusters = 4
    elif type == "standardization":
        clusters = 6
    
    agglomerator = FeatureAgglomeration(
        n_clusters=clusters,
        metric="euclidean",
        linkage="ward"
    )
    
    X_train_agg = agglomerator.fit_transform(X_train)
    X_test_agg = agglomerator.transform(X_test)
    
    return X_train_agg, X_test_agg

def undersampling(X_train, y_train):
    cc = ClusterCentroids(random_state=21)
    
    X_train_und, y_train_und = cc.fit_resample(X_train, y_train)
    
    return X_train_und, y_train_und

def oversampling(X_train, y_train):
    smote = SMOTE(random_state=21)

    X_train_und, y_train_und = smote.fit_resample(X_train, y_train)
    
    return X_train_und, y_train_und

def combine_preprocessing(X, y, techniques):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

    if "standardization" in techniques:
        X_train, X_test = standardization(X_train, X_test)
    elif "normalization" in techniques:
        X_train, X_test = normalization(X_train, X_test)
    
    if "agglomeration" in techniques:
        if "standardization" in techniques:
            X_train, X_test = agglomeration(X_train, X_test, "standardization")
        elif "normalization" in techniques:
            X_train, X_test = agglomeration(X_train, X_test, "normalization")

    if "undersampling" in techniques:
        X_train, y_train = undersampling(X_train, y_train)
    elif "oversampling" in techniques:
        X_train, y_train = oversampling(X_train, y_train)
    
    return X_train, X_test, y_train, y_test