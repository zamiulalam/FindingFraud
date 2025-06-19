import pandas as pd
import os, gc, sys
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import roc_curve,auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score

import argparse
sys.path.append("../source/")
from helper_functions import *


def loadData(transaction_path="train_transactions.csv", ID_path=None):

    dl = DataLoader()
    df = dl.load_csv(transaction_path)

    if ID_path:
        df.transaction = False
        df_id = dl.load_csv(ID_path)

        df = df.merge(df_id, how='left', left_index=True, right_index=True)


def prune_columns(df:pd.DataFrame, columns:list, keep:bool=True):
    if keep:
        return df[columns]
    else:
        df = df.drop(columns=columns)

def plot_roc(y_true, y_prob):
     
    # Compute ROC curve and AUC
    fpr, tpr, threshold = roc_curve(y_true.astype(int).values, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("roc_auc.png")

    return optimal_threshold

def plot_pr(y_true, y_prob):
    # Compute ROC curve and AUC
    precision, recall, threshold = precision_recall_curve(y_true.astype(int).values, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_index = np.argmax(f1_scores)
    optimal_threshold = threshold[best_index]

    roc_auc = auc(recall, precision)
    print(f"Best F1: {f1_scores[best_index]:.4f} at threshold: {optimal_threshold:.4f}")
    print("auc", roc_auc)
    plt.figure()
    plt.plot(recall,precision, label=f'ROC curve (AUPRC = {roc_auc:.2f})')
    plt.plot([0, 1], [1, 0], 'k--')  # Diagonal line
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("precision_recall.png")

    return optimal_threshold


def main():
    import warnings
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--transaction', type=str, default='train_transactions.csv', help="path to transaction csv")
    parser.add_argument('-i', '--id', type=str, default=None, help='path to identity csv. Defaults to None')
    parser.add_argument('--tts', type=float, default=-1, help="Train test split. Default is to train on all data")
    parser.add_argument('--tr_cols', type=str, default=None, help="Text file containing a list of variables to keep in transaction dataframe")
    parser.add_argument('--id_cols', type=str, default=None, help="Text file containing a list of variables to keep in identity dataframe")

    # parser.add_argument('-p', '--prune', type=str, default=None, help="Text file containing a list of variables to remove from dataframe")
    # parser.add_argument('-k', '--keep', type=str, default=None, help="Text file containing a list of variables to keep in dataframe")
    
    args = parser.parse_args()
    dataLoader = DataLoader()

    df = dataLoader.load_csv(args.transaction, args.id, args.tr_cols, args.id_cols)
    
    y = df["isFraud"]
    # X = df.drop(columns=["isFraud", "uid", "card1", "addr1", "D1", "TransactionID", "D1n"]) # drop items that go into uid?
    if 'uid' in set(df.columns):
        X = df.drop(columns=["isFraud", "uid", "TransactionID"])
    else:
        X = df.drop(columns=["isFraud", "TransactionID"])

    params = {"device":"cuda", "objective":"binary:logistic",
                "eval_metric":"aucpr", #logloss
                'learning_rate': 0.12,
                'max_depth': 12, }

    if args.tts:
        from sklearn.model_selection import train_test_split
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        dtrain = xgb.DMatrix(X_train, label=y_train) 
        dval = xgb.DMatrix(X_test, label=y_test)
        y_prob = model.predict(dval)
    else:
        dtrain = xgb.DMatrix(X, label=y) 


    model = xgb.train(params, dtrain, num_boost_round=100)
    importance = model.get_score(importance_type='gain')

    model.save_model("model.json")

    if args.tts:
        plot_roc(y_test, y_prob)
        plot_pr(y_test, y_prob)

if __name__ == '__main__':
    main()