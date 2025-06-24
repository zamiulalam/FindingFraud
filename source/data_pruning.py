import pandas as pd
import numpy as np


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def remove_correlated_columns(df, columns, target_col="isFraud", keep_corr=True, corr_factor=0.9):
    """
    df:         pandas dataframe
    column:     list of column names to consider
    target:     the column who's final correlation we are interested in
    keep_corr:  keep only the column which is most correlated with the target. If False, keep the column with the most unique values
    corr_factor how high correlated variables must be to consider them
    return:     List of variables to drop
    """

    # Step 1: Compute correlation matrix for input features
    corr_matrix = df[columns].corr().abs()

    # Step 2: Mask upper triangle and self-correlations
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    high_corr_pairs = corr_matrix.where(upper).stack()
    high_corr_pairs = high_corr_pairs[high_corr_pairs > corr_factor]

    # Step 3: Determine which column in each pair to drop
    to_drop = set()
    for col1, col2 in high_corr_pairs.index:

        if keep_corr:
            corr1 = abs(df[[col1, target_col]].corr().iloc[0, 1])
            corr2 = abs(df[[col2, target_col]].corr().iloc[0, 1])
            drop_col = col1 if corr1 < corr2 else col2
        else:
            drop_col = col1 if df[col1].nunique() >= df[col2].nunique() else col2

        to_drop.add(drop_col)

    return to_drop



def get_low_importance_v_features(train, threshold=20):
    vfeatures = [f for f in train.columns if f.startswith("V")]

    # Impute missing values using medians from train
    train_v = train[vfeatures].fillna(train[vfeatures].median())

    # Train-validation split
    split = int(0.8 * len(train))
    V_x, V_cv = train_v[:split], train_v[split:]
    y_train = train["isFraud"][:split]
    y_cv = train["isFraud"][split:]

    # Train LightGBM
    clf = LGBMClassifier()
    clf.fit(V_x, y_train)

    # AUC scores
    print("Train AUC:", roc_auc_score(y_train, clf.predict_proba(V_x)[:, 1]))
    print("CV AUC:", roc_auc_score(y_cv, clf.predict_proba(V_cv)[:, 1]))

    # Collect low-importance features
    Vremove = [vfeatures[i] for i, imp in enumerate(clf.feature_importances_) if imp < threshold]
    print(f"Dropping {len(Vremove)} V-features with importance < {threshold}")

    return Vremove
