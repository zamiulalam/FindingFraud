import pandas as pd
import numpy as np
import dcor

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def remove_correlated_columns(df: pd.DataFrame, columns: list[str], target_col: str = "isFraud", keep_corr: bool = True)->set[str]:
    """Returns a set of variables to drop

    Parameters
    -----------------------------------
    df: pd.DataFrame
        The pandas dataframe to be considered for correlation analysis.
    column: list[str]
        The list of column names to be considered.
    target_col: str 
        The name of the column whose correlation with other columns will determine which column to retain. 
    keep_corr: bool
        Flag to keep only the column which is the most correlated column with the target. If False, keep the column with the most unique values.

    Returns
    -----------------------------------
    set[str]
        A set of columns that are to be dropped.
    """

    # Step 1: Compute correlation matrix for input features
    corr_matrix = df[columns].corr().abs()

    # Step 2: Mask upper triangle and self-correlations
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    high_corr_pairs = corr_matrix.where(upper).stack()
    high_corr_pairs = high_corr_pairs[high_corr_pairs > 0.85]

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

def get_low_importance_v_features(train: pd.DataFrame, threshold: float = 20, verbose: bool = False)->list[str]:
    """Returns a list of V columns to remove

    Parameters
    -----------------------------------
    train: pd.DataFrame
        The pandas dataframe that is to be used for determining which features to drop. 
    threshold: float
        The threshold for feature importance. If the feature importance of a feature is below the threshold, it is added to the list of columns to remove.
    verbose: bool
        A flag to toggle on printing of information about the number of variables that are removed.

    Returns
    -----------------------------------
    list[str]
        A list of features that are to be removed.
    """
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
    if verbose:
        print(f"Dropping {len(Vremove)} V-features with importance < {threshold}")

    return Vremove

def remove_cols_w_high_NaN_percentage(df: pd.DataFrame, col_list: list[str]|None = None, threshold: float = 0.8)->set[str]:
    """Returns a list of columns to remove from a pandas dataframe based on high NaN percentage 
    Parameters
    -----------------------------------
    df: pd.DataFrame
        The pandas dataframe that is being evaluated.
    col_list: list[str]
        The list of columns to be considered. If not supplied, all columns of the dataframe will be considered by default.
    threshold: float
        The threshold to be used for removal of columns. This is the minimum percentage of the values that should be NaN in order to be considered for removal. By default the threshold is 0.8 (80%).

    Returns
    -----------------------------------
    set[str]
        The set of columns to be removed.
    """
    if col_list is None:
        col_list = df.columns
    to_remove = set()
    for col in col_list:
        nan_percentage = df[col].isna().sum()/df[col].shape[0]
        if nan_percentage > threshold:
            to_remove.add(col)
    return to_remove

def get_highly_distance_correlated_columns(df: pd.DataFrame, target_col: str = 'isFraud', threshold: float = 0.4)->dict:
    """Returns a list of features that are highly correlated with target column. 

    Parameters
    -----------------------------------
    df: pd.DataFrame
        The pandas dataframe whose columns are to be considered for distance correlation evaluation. 
    target_col: str
        The target column with which the distance correlation is to be computed.
    threshold: float
        The threshold of distance correlation above which the columns are retained. 

    Returns
    -----------------------------------
    dict
        A dictionary of column names as keys and distance correlation as values. The keys are columns to be retained based on high distance correlation with the target column. 
    """
    high_corr_features = {}
    for col in df.columns:
        # Remove the NaN values for distance correlation computation
        mask = np.logical_not(df[col].isna())
        nan_dropped_col = df[col][mask]
        if df[col].dtype == 'object':
            le = LabelEncoder()
            nan_dropped_col = le.fit_transform(nan_dropped_col)
        disco = dcor.distance_correlation(nan_dropped_col, df[target_col][mask])
        if disco > threshold:
            high_corr_features[col] = disco

    # Sort based on descending order of distance correlation
    high_corr_features = dict(sorted(high_corr_features.items(), key = lambda item: item[1], reverse= True))
    return high_corr_features

def remove_highly_correlated_features_w_same_NaN_num(df: pd.DataFrame, high_corr_features: dict, verbose: bool = False)->list[str]:
    """Returns a list of features to retain. Inspired by: https://www.kaggle.com/code/cdeotte/eda-for-columns-v-and-id

    Parameters
    -----------------------------------
    df: pd.DataFrame
        The pandas dataframe whose columns are to be considered.
    high_corr_features: dict
        A dictionary whose keys are the columns with high distance correlation with the target column and the values are the distance correlation with the target column.
    verbose: bool
        Flag that toggles on printing verbose information about which columns have same number of NaNs and which of them are to be retained. 

    Returns
    -----------------------------------
    list[str]
        The list of features that are to be retained for training.
    """
    df_high_corr_features_isnan = df[list(high_corr_features.keys())].isna()
    sim_nan_num = dict()
    for col in high_corr_features.keys():
        nan_count = df_high_corr_features_isnan[col].sum()
        try:
            sim_nan_num[nan_count].append(col)
        except:
            sim_nan_num[nan_count] = [col]
    sim_nan_num = dict(sorted(sim_nan_num.items()))

    retain_list = []
    for values in sim_nan_num.values():
        max_disco = high_corr_features[values[0]]
        feature_to_retain = values[0]
        for each_value in values:
            if high_corr_features[each_value] > max_disco:
                feature_to_retain = each_value
        retain_list.append(feature_to_retain)
    if verbose:
        for key, value in sim_nan_num.items():
            print(f"columns {value} have same number of NaNs: {key}")
        print("-----------------------\n Retaining following features:")
        print(retain_list)

    return retain_list
