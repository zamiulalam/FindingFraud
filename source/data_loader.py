from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from os.path import join as join_path
import numpy as np

class DataLoader():
    def __init__(self, predict:bool=False):
        """
        predict: will data be used for prediction? If False, it is assumed to be used for training
        """
        self._data_path = ""
        self._predict = predict
        self.df = None

    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, value):
        self._df = value
    
    def add_transaction_features(self):
        """
        Add useful features and factorize categorical features
        """
        self.df["TransactionDay"] = np.floor(self.df["TransactionDT"] / (24*60*60))

        for i in range(1,10):
            self.df["M"+str(i)] = self.df["M"+str(i)].astype(bool)

        # Deal with categorical features
        # trees don't care about cardinality, we can transform them all into integer codes. Some such as addr1 are already numbers
        #categorical_vars = ["ProductCD", "card1","card2","card3","card4","card5","card6","addr1", "addr2", "P_emaildomain", "R_emaildomain"]
        categorical_vars = ["ProductCD","card4","card6","P_emaildomain", "R_emaildomain"]
        for cat in categorical_vars:
            self.df[cat] = pd.factorize(self.df[cat])[0]
        
    def add_uid(self):
        """
        add universal id to dataset
        """
        self.df["D1n"] = self.df["TransactionDay"] - self.df["D1"]
        self.df['uid'] = self.df["card1"].astype(str)+'_'+self.df["addr1"].astype(str)+'_'+self.df["D1n"].astype(str)

    def transaction_in_window(self):
        """
        Determines how many transactions of the same amount occur in plus/minus 500 seconds around a given transaction
        """
        # Sort by TransactionTD
        self.df = self.df.sort_values('TransactionDT').reset_index(drop=True)

        # Create an empty column
        self.df['IsDuplicateInWindow'] = 1

        # Get numpy arrays for faster operations
        tds = self.df['TransactionDT'].values
        amts = self.df['TransactionAmt'].values

        # Use searchsorted to find window ranges
        for i in range(len(self.df)):
            lower = tds[i] - 500
            upper = tds[i] + 500

            # Find indices where TD is within ±100
            start = np.searchsorted(tds, lower, side='left')
            end = np.searchsorted(tds, upper, side='right')

            # Slice the relevant window and check for other matching TransactionAmt
            window_amts = amts[start:end]
            match_count = np.sum(window_amts == amts[i])

            self.df.at[i, 'IsDuplicateInWindow'] = match_count
        
    
    def reduce_mem_usage(self):
        """
        Downcast all object types to smallest type able to represent the range in the dataset
        """

        for col in self.df.select_dtypes(include=[np.number]).columns:
            col_data = self.df[col]    

            if pd.api.types.is_float_dtype(col_data):
                # Downcast to float
                self.df[col] = pd.to_numeric(col_data, downcast='float')
            elif pd.api.types.is_integer_dtype(col_data):
                # Downcast to smaller int
                self.df[col] = pd.to_numeric(col_data, downcast='integer')

        self.df.info(memory_usage='deep')

    def encode_AG(self, groupby:str, aggregate_cols:list):

        # Compute the mean and std only for those columns
        means = self.df.groupby(groupby)[aggregate_cols].transform('mean').add_suffix('_uid_mean')
        stds = self.df.groupby(groupby)[aggregate_cols].transform('std').add_suffix('_uid_std')

        # Concatenate the results with the original dataframe
        self.df = pd.concat([self.df, means, stds], axis=1)

    def encode_FE(self, cols:list, inplace=False):
        """
        input:
        cols: list of features to frequency encode
        inplace: replace the columns with their frequency encoding. 
        output:
        dataframe with frequency encoded features
        """
        # cols = str or list of str
        if isinstance(cols, str):
            cols = [cols]

        
        for col in cols:
            if inplace:
                col_name = col
            else:
                col_name = col+"_FE"
            freq = self.df[col].value_counts(normalize=True)
            self.df[col_name] = self.df[col].map(freq)

    
    def iterative_imputation(self, max_iter=10, random_state=42):
        """
        Impute missing values in self.df using IterativeImputer.
        Assumes all columns are numeric or already encoded.
        """
        columns = self.df.columns
        index = self.df.index

        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        imputed_array = imputer.fit_transform(self.df)

        self.df = pd.DataFrame(imputed_array, columns=columns, index=index)
    
    
    def load_csv(self, transaction_file=None, identity_file=None, 
                 tr_columns:str=None, id_columns:str=None):
        
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
            if 'TransactionID' not in usecols:
                usecols.append('TransactionID')
            if not self._predict:
                if 'isFraud' not in usecols:
                    usecols.append('isFraud')
            else:
                if 'isFraud' in usecols:
                    usecols.remove('isFraud')

        # Open transaction file
        with open(join_path(self._data_path, transaction_file)) as f:

            if tr_columns:
                self.df = pd.read_csv(f, usecols=usecols)
            else:
                 self.df = pd.read_csv(f)

        self.reduce_mem_usage()

        # Read in list of ID columns to use
        if id_columns:
            usecols = []
            with open(id_columns) as f:
                usecols = f.readlines()
            usecols = [item.strip() for item in usecols]
            if 'TransactionID' not in usecols:
                usecols.append('TransactionID')

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
                self.df = self.df.merge(id_df, on="TransactionID", how='left')
                del id_df
       
        self.reduce_mem_usage()
        return self.df
