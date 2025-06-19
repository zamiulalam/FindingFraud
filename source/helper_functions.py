import pandas as pd
from os.path import join as join_path
import numpy as np

class DataLoader():
    def __init__(self, transaction=True):
        self._data_path = "../../data/"
        self._transaction = transaction
        self._df = None

    @property
    def transaction(self):
        return self._transaction
    
    @transaction.setter
    def transaction(self, value):
        self._transaction = value
    
    def add_transaction_features(self):
        """
        Add useful features and factorize categorical features
        """
        self._df["TransactionDay"] = np.floor(self._df["TransactionDT"] / (24*60*60))

        for i in range(1,10):
            self._df["M"+str(i)] = self._df["M"+str(i)].astype(bool)

        # Deal with categorical features
        # trees don't care about cardinality, we can transform them all into integer codes. Some such as addr1 are already numbers
        #categorical_vars = ["ProductCD", "card1","card2","card3","card4","card5","card6","addr1", "addr2", "P_emaildomain", "R_emaildomain"]
        categorical_vars = ["ProductCD","card4","card6","P_emaildomain", "R_emaildomain"]
        for cat in categorical_vars:
            self._df[cat] = pd.factorize(self._df[cat])[0]
            # df[cat] =  self._label_encoder.fit_transform(df[cat])
    
        return self._df
    
    def add_uid(self):
        """
        add universal id to dataset
        """
        self._df["D1n"] = self._df["TransactionDay"] - self._df["D1"]
        self._df['uid'] = self._df["card1"].astype(str)+'_'+self._df["addr1"].astype(str)+'_'+self._df["D1n"].astype(str)

    def transaction_in_window(self):
        """
        Determines how many transactions of the same amount occur in plus/minus 500 seconds around a given transaction
        """
        # Sort by TransactionTD
        self._df = self._df.sort_values('TransactionDT').reset_index(drop=True)

        # Create an empty column
        self._df['IsDuplicateInWindow'] = 1

        # Get numpy arrays for faster operations
        tds = self._df['TransactionDT'].values
        amts = self._df['TransactionAmt'].values

        # Use searchsorted to find window ranges
        for i in range(len(self._df)):
            lower = tds[i] - 500
            upper = tds[i] + 500

            # Find indices where TD is within ±100
            start = np.searchsorted(tds, lower, side='left')
            end = np.searchsorted(tds, upper, side='right')

            # Slice the relevant window and check for other matching TransactionAmt
            window_amts = amts[start:end]
            match_count = np.sum(window_amts == amts[i])

            self._df.at[i, 'IsDuplicateInWindow'] = match_count
    
    def encode_features(self, columns=None):
        """
        Aggregate encoding and frequency encoding of variables
        """
    
        columns_to_encode = []
        # columns_to_encode = [v for v in df.columns if v.startswith("V")]
        columns_to_encode.append("TransactionAmt")
        columns_to_encode.append("TransactionDT")

        d_columns = [d for d in self._df.columns if d.startswith("D") and len(d) < 4]

        columns_to_encode += d_columns
        self._df = encode_AG(self._df, 'uid', columns_to_encode)

        columns_to_encode = ["addr1", "card1", "card2", "card3", "P_emaildomain", "R_emaildomain"]
        self._df = encode_FE(self._df, columns_to_encode)
    
    def reduce_mem_usage(self):
        """
        Downcast all object types to smallest type able to represent the range in the dataset
        """

        for col in self._df.select_dtypes(include=[np.number]).columns:
            col_data = self._df[col]    

            if pd.api.types.is_float_dtype(col_data):
                # Downcast to float
                self._df[col] = pd.to_numeric(col_data, downcast='float')
            elif pd.api.types.is_integer_dtype(col_data):
                # Downcast to smaller int
                self._df[col] = pd.to_numeric(col_data, downcast='integer')

        self._df.info(memory_usage='deep')

    
    def load_csv(self, transaction_file=None, identity_file=None, 
                 tr_columns:str=None, id_columns:str=None,
                 tr_features=True, uid=True, tr_window=True, encode=True):
        
        """
        transaction_file:   path to transaction csv
        identity_file:      path to identity file. Optional. If included it will be merged with transaction dataframe
        tr_columns:         path to file containing names of all transaction columns to include. If left blank all columns will be loaded
        id_columns:         path to file containing names of all id columns to include. If left blank all columns will be loaded

        returns: None. Modifies dataframe in place
        """

        # Read in list of columns to use
        if tr_columns:
            usecols = []
            with open(tr_columns) as f:
                usecols = f.readlines()
            usecols = [item.strip() for item in usecols]

        # Open transaction file
        with open(join_path(self._data_path, transaction_file)) as f:

            if tr_columns:
                self._df = pd.read_csv(f, usecols=usecols)
            else:
                 self._df = pd.read_csv(f)

        self.reduce_mem_usage()

        # Read in list of ID columns to use
        if id_columns:
            usecols = []
            with open(id_columns) as f:
                usecols = f.readlines()
            usecols = [item.strip() for item in usecols]

        # Read ID file
        if identity_file:
            with open(join_path(self._data_path, identity_file)) as f:
                if id_columns:
                    id_df = pd.read_csv(f, usecols=usecols)
                else:
                    id_df = pd.read_csv(f)
                
                # convert objects to ints
                for col in id_df.select_dtypes(include='object').columns:
                    id_df[col], _ = pd.factorize(id_df[col])
                
                # merge ID dataframe with transaction df
                self._df = self._df.merge(id_df, on="TransactionID", how='left')
                del id_df

        if self._transaction:
            if tr_features:    
                self.add_transaction_features()
                print("done")
            if uid:
                self.add_uid()
                print("uid")
            if tr_window:
                self.transaction_in_window()
                print("what")
            if encode:
                self.encode_features()
       
        self.reduce_mem_usage()
        return self._df
    
def remove_correlated_columns(df, columns, target_col="isFraud", keep_corr=True):
    """
    df:         pandas dataframe
    column:     list of column names to consider
    target:     the column who's final correlation we are interested in
    keep_corr:  keep only the column which is most correlated with the target. If False, keep the column with the most unique values
    return:     List of variables to drop
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


def encode_AG( df:pd.DataFrame,
              groupby:str, 
              aggregate_cols:list
             
):

    print(aggregate_cols)  
    # Compute the mean and std only for those columns
    means = df.groupby(groupby)[aggregate_cols].transform('mean').add_suffix('_uid_mean')
    stds = df.groupby(groupby)[aggregate_cols].transform('std').add_suffix('_uid_std')

    # Concatenate the results with the original dataframe
    df = pd.concat([df, means, stds], axis=1)
    return df

def encode_FE(df, cols):
    # cols = str or list of str
    if isinstance(cols, str):
        cols = [cols]

    
    for col in cols:
        col_name = col+"_FE"
        freq = df[col].value_counts(normalize=True)
        df[col_name] = df[col].map(freq)

    return df