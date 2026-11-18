# src/data_loader/splitter.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def chronological_split_and_scale(prices, features, train_ratio=0.70, val_ratio=0.15):
    common_idx = prices.index.intersection(features.index)
    prices = prices.loc[common_idx]
    features = features.loc[common_idx]
    
    n = len(common_idx)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Kronolojik Dilimleme
    prices_train, features_train = prices.iloc[:train_end], features.iloc[:train_end]
    prices_val, features_val = prices.iloc[train_end:val_end], features.iloc[train_end:val_end]
    prices_test, features_test = prices.iloc[val_end:], features.iloc[val_end:]
    
    # Sızıntısız Ölçeklendirme
    scaler = StandardScaler()
    
    scaled_features_train = pd.DataFrame(
        scaler.fit_transform(features_train), 
        index=features_train.index, 
        columns=features_train.columns
    )
    
    scaled_features_val = pd.DataFrame(
        scaler.transform(features_val), 
        index=features_val.index, 
        columns=features_val.columns
    )
    
    scaled_features_test = pd.DataFrame(
        scaler.transform(features_test), 
        index=features_test.index, 
        columns=features_test.columns
    )
    
    return (
        (prices_train, scaled_features_train),
        (prices_val, scaled_features_val),
        (prices_test, scaled_features_test)
    )