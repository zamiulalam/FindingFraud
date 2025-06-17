import pandas as pd
from os.path import join as join_path
import numpy as np

class DataLoader():
    def __init__(self, transaction=True):
        self._data_path = "../../data/"
        self._transaction = transaction

    @property
    def transaction(self):
        return self._transaction
    @transaction.setter
    def transaction(self, value):
        self._transaction = value
    
    def add_transaction_features(self, df):
        df["TransactionDay"] = df["TransactionDT"] / (24*60*60)
        d_columns = [f'D{i}' for i in range(1, 16)]
        for i, col in enumerate(d_columns):
            df[col] = df["TransactionDay"] - df[col]

        for i in range(1,10):
            df["M"+str(i)] = df["M"+str(i)].astype(bool)
        # Deal with categorical features
        # trees don't care about cardinality, we can transform them all into integer codes. Some such as addr1 are already numbers
        #categorical_vars = ["ProductCD", "card1","card2","card3","card4","card5","card6","addr1", "addr2", "P_emaildomain", "R_emaildomain"]
        categorical_vars = ["ProductCD","card4","card6","P_emaildomain", "R_emaildomain"]
        for cat in categorical_vars:
            df[cat] = pd.factorize(df[cat])[0]
            # df[cat] =  self._label_encoder.fit_transform(df[cat])
    
        return df
    
    def reduce_mem_usage(self, df):

        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col]    

            if pd.api.types.is_float_dtype(col_data):
                # Downcast to float
                df[col] = pd.to_numeric(col_data, downcast='float')
            elif pd.api.types.is_integer_dtype(col_data):
                # Downcast to smaller int
                df[col] = pd.to_numeric(col_data, downcast='integer')

        df.info(memory_usage='deep')
        return df
    
    def aggregate_encoding(self, df):
        return

    def load_csv(self, file_name):
        with open(join_path(self._data_path, file_name)) as f:
            df = pd.read_csv(f)#, nrows=200000)

        

        if self._transaction:    
            self.add_transaction_features(df)

        df = self.reduce_mem_usage(df)

        return df
    

def remove_correlated_columns(df, columns, target_col="isFraud", keep_corr=True):
    """
    df:         pandas dataframe
    column:     list of column names to consider
    target:     the column who's final correlation we are interested in
    keep_corr:  keep only the column which is most correlated with the target. If False, keep the column with the most unique values
    return:     new dataframe with columns removed if they are highly correlated. Only the column most correlated with the target is kept
    """

    # Step 1: Compute correlation matrix for input features
    corr_matrix = df[columns].corr().abs()

    # Step 2: Mask upper triangle and self-correlations
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    high_corr_pairs = corr_matrix.where(upper).stack()
    high_corr_pairs = high_corr_pairs[high_corr_pairs > 0.95]

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

    # Step 4: Drop selected columns
    df = df.drop(columns=list(to_drop))

    return df

def encode_AG( df:pd.DataFrame,
              groupby:str, 
              aggregate_cols:list
             
):
    
    # Compute the mean and std only for those columns
    means = df.groupby(groupby)[aggregate_cols].transform('mean').add_suffix('_uid_mean')
    stds = df.groupby(groupby)[aggregate_cols].transform('std').add_suffix('_uid_std')

    # Concatenate the results with the original dataframe
    df = pd.concat([df, means, stds], axis=1)
    return df
