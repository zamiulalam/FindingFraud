import pandas as pd
import os, gc, sys
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import roc_curve,auc, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score

from source.data_loader import *

def get_path(file_name):
    models_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.dirname(models_dir)
    models_dir = os.path.join(models_dir, file_name)
    return models_dir

def plot_roc(y_true, y_prob, name=""):
     
    # Compute ROC curve and AUC
    fpr, tpr, threshold = roc_curve(y_true.astype(int).values, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    roc_auc = auc(fpr, tpr)
    print("roc auc", roc_auc, "at threshold",optimal_threshold)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    output_path = get_path(f"outputs/roc_{name}.png")
    plt.savefig(output_path)
    return optimal_threshold

def plot_pr(y_true, y_prob, name=""):
    # Compute ROC curve and AUC
    precision, recall, threshold = precision_recall_curve(y_true.astype(int).values, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_index = np.argmax(f1_scores)
    optimal_threshold = threshold[best_index]

    roc_auc = auc(recall, precision)
    print(f"Best F1: {f1_scores[best_index]:.4f} at threshold: {optimal_threshold:.4f}")
    print("auprc", roc_auc)
    plt.figure()
    plt.plot(recall,precision, label=f'ROC curve (AUPRC = {roc_auc:.2f})')
    plt.plot([0, 1], [1, 0], 'k--')  # Diagonal line
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)

    output_path = get_path(f"outputs/precision_recall_{name}.png")
    plt.savefig(output_path)

    return optimal_threshold

def plot_confusion_matrix(y_true, y_pred, threshold, name=""):
    y_pred = y_pred > threshold
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    output_path = get_path(f"outputs/confusion_matrix_{name}.png")
    disp.plot(cmap='Blues').figure_.savefig(output_path)

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--transaction', type=str, default='train_transactions.csv', help="path to transaction csv")
    parser.add_argument('-i', '--id', type=str, default=None, help='path to identity csv. Defaults to None')
    parser.add_argument('-m', '--model', type=str, default='model.json', help="Name of input/output model including file extension. (i.e model.json)")
    parser.add_argument('--train_test_split', type=float, default=-1, help="Train test split. Default is to train on all data")
    parser.add_argument('--tr_cols', type=str, default=None, help="Text file containing a list of variables to keep in transaction dataframe")
    parser.add_argument('--id_cols', type=str, default=None, help="Text file containing a list of variables to keep in identity dataframe")
    parser.add_argument('--predict', action='store_true', help="Make predictions instead of training")

    # parser.add_argument('-p', '--prune', type=str, default=None, help="Text file containing a list of variables to remove from dataframe")
    # parser.add_argument('-k', '--keep', type=str, default=None, help="Text file containing a list of variables to keep in dataframe")
    
    args = parser.parse_args()
    dataLoader = DataLoader(args.predict)

    dataLoader.load_csv(args.transaction, args.id, args.tr_cols, args.id_cols)

    # add additional features
    dataLoader.add_transaction_features()
    dataLoader.add_uid()
    dataLoader.transaction_in_window()

    # aggregate and frequency encoding features
    columns_to_encode = []
    columns_to_encode.append("TransactionAmt")
    columns_to_encode.append("TransactionDT")

    d_columns = [d for d in dataLoader.df.columns if d.startswith("D") and len(d) < 4]

    columns_to_encode += d_columns
    dataLoader.encode_AG('uid', columns_to_encode)

    columns_to_encode = ["addr1", "card1", "card2", "card3", "P_emaildomain", "R_emaildomain"]
    dataLoader.encode_FE(columns_to_encode)

    df = dataLoader.df
    
    if args.predict:
        TransactionID = df['TransactionID']
        df = df.drop(columns=['TransactionID', 'uid'])
        dval = xgb.DMatrix(df)
        model = xgb.Booster()
        model_path = get_path(f"models/{args.model}")
        model.load_model(model_path)
        y_prob = model.predict(dval)

        output_df = pd.DataFrame(data=zip(TransactionID,y_prob), columns=["TransactionID", "isFraud"])
        output_path = get_path(f"outputs/predictions_{args.model}.csv")
        output_df.to_csv(output_path, index=False)

        return


    # Prepare inputs and targets for training
    df = df.sort_values(by='TransactionDT', ascending=True)
    y = df["isFraud"]
    if 'uid' in set(df.columns):
        X = df.drop(columns=["isFraud", "uid", "TransactionID"])
    else:
        X = df.drop(columns=["isFraud", "TransactionID"])

    params = {"device":"cuda", "objective":"binary:logistic",
                "eval_metric":"aucpr", #logloss
                'learning_rate': 0.1,
                'max_depth': 16, }

    if args.train_test_split > 0:
        from sklearn.model_selection import train_test_split
        
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.train_test_split, random_state=402, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.train_test_split, stratify=None, shuffle=False) # train and test using temporally separated data

        dtrain = xgb.DMatrix(X_train, label=y_train) 
        dval = xgb.DMatrix(X_test, label=y_test)
    else:
        dtrain = xgb.DMatrix(X, label=y) 


    model = xgb.train(params, dtrain, num_boost_round=100)
    importance = model.get_score(importance_type='gain')

    models_dir = get_path(f"models/{args.model}")
    print("Saving model to ",models_dir)

    model.save_model(models_dir)

    if args.train_test_split > 0:
        y_prob = model.predict(dval)
        opt_threshold = plot_roc(y_test, y_prob, args.model)
        plot_pr(y_test, y_prob, args.model)
        plot_confusion_matrix(y_test, y_prob, opt_threshold, args.model)

if __name__ == '__main__':

    main()