from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from os.path import join as join_path
import numpy as np
from dataclasses import dataclass
from typing import Optional
@dataclass
class TextColor:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

class DataLoader():
    def __init__(self):
        self.df: pd.DataFrame|None = None
        self.float_cols: list[str] = []
        self.int_cols: list[str] = []
        self.str_cols: list[str] = []
        self.bool_cols: list[str] = []


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

    def encode_AG(self, groupby: str, aggregate_cols: list[str])->None:
        """Add new mean and standard deviation columns for a given set of columns

        Parameters
        -----------------------------------
        groupby: str
            The column that is used to group the dataframe. 
        aggregate_cols: list[str]
            The list of columns for which the aggregate values are to computed.

        Returns
        -----------------------------------
        None
            The dataframe is modified to add new columns for mean and standard deviation of the columns to be aggregated.
        """

        # Compute the mean and std only for those columns
        means = self.df.groupby(groupby)[aggregate_cols].transform('mean').add_suffix('_uid_mean')
        stds = self.df.groupby(groupby)[aggregate_cols].transform('std').add_suffix('_uid_std')

        # Concatenate the results with the original dataframe
        self.df = pd.concat([self.df, means, stds], axis=1)

    def encode_FE(self, cols: list[str], inplace: bool = False)->None:
        """Modifies the pandas dataframe with frequency encoding either inplace or by adding new columns.

        Parameters
        -----------------------------------
        cols: list[str] 
            List of features to frequency encode
        inplace: bool
            A flag which when set to True replace the columns with their frequency encoding. 
    
        Returns
        -----------------------------------
        None
            The dataframe is modified with frequency encoded features
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

    def iterative_imputation(self, max_iter: int = 10, random_state: int = 42)->None:
        """Impute missing values in self.df using IterativeImputer.Assumes all columns are numeric or already encoded.

        Parameters
        -----------------------------------
        max_iter: int
            The maximum number of iterations to use for the imputer
        random_state: int
            The random state to be used by the imputer. Set it to get reproducible results

        Returns
        -----------------------------------
        None
            Modified pandas dataframe object in place
        """
        columns = self.df.columns
        index = self.df.index

        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        imputed_array = imputer.fit_transform(self.df)

        self.df = pd.DataFrame(imputed_array, columns=columns, index=index)
    
    
    def load_csv(self, transaction_file: str|None = None, identity_file: Optional[str] = None, 
                 tr_columns: str|None = None, id_columns: str|None = None,
                 isTest: bool = False)->None:
        
        """
        Parameters
        -----------------------------------
        transaction_file: str|None
            Path to transaction csv
        identity_file: Optional[str]
            Path to identity file. Optional. If included it will be merged with transaction dataframe
        tr_columns: str|None
            Path to file containing names of all transaction columns to include. If left blank all columns will be loaded
        id_columns: str|None
            Path to file containing names of all id columns to include. If left blank all columns will be loaded
        isTest: bool
            A flag to specify whether the csv being loaded is test sample or train sample

        Returns
        -----------------------------------
        None 
            Modifies dataframe in place
        """

        # Read in list of columns to use
        if tr_columns:
            usecols = []
            with open(tr_columns) as f:
                usecols = f.readlines()
            usecols = [item.strip() for item in usecols]
        if not isTest and 'isFraud' not in usecols:
            print(f"{TextColor.YELLOW}The columns supplied do not have isFraud column which is needed for training, adding it{TextColor.RESET}")
            usecols.append('isFraud')
        elif isTest and 'isFraud' in usecols:
            usecols.remove('isFraud')

        # Open transaction file
        with open(transaction_file) as f:

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
                print(f"{TextColor.YELLOW}The ID columns do not have TransactionID column. Adding it for merging dataframes{TextColor.RESET}")
                usecols.append("TransactionID")

        # Read ID file
        if identity_file:
            with open(identity_file) as f:
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

    def getListOfColTypes(self):
        """
        Adds the columns of a certain data type into corresponding list so that it is easily accessible. Useful for encoding, for example. 

        Only call this function after the load_csv method has been called on an instance of the class.
        """
        if self.df is None:
            print(f"{TextColor.RED}The dataframe is None! Please make sure you have called load_csv method first and got a valid pandas dataframe. Returning empty lists{TextColor.RESET}")
            return
        for col, col_type in self.df.dtypes.to_dict().items():
            if col_type in ['float16', 'float32', 'float64']:
                self.float_cols.append(col)
            elif col_type in ['int8', 'int16', 'int32', 'int64']:
                self.int_cols.append(col)
            elif col_type == 'object':
                self.str_cols.append(col)
            elif col_type == 'bool':
                self.bool_cols.append(col)
            else:
                print(f"column {col} is not one of the [float64, int64, object, bool] types. It's type is {col_type} ")
