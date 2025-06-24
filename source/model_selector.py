import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class ModelSelector:
    def __init__(self, model_type:str='xgboost', use_gpu:bool=False, test_size:float=0.2, random_state:int=42, **model_params):
        """
        Initialize the ModelSelector with a specified model.

        Parameters:
        - model_type (str): 'xgboost', 'lightgbm', or 'catboost'
        - use_gpu (bool): whether to enable GPU acceleration
        - test_size (float): fraction of data to use for validation
        - random_state (int): seed for reproducibility
        - model_params: additional keyword arguments to pass to the model
        """
        self.model_type = model_type.lower()
        self.use_gpu = use_gpu
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params
        self.model = None

    def _get_model(self):
        if self.model_type == 'xgboost':
            default_params = {
                'use_label_encoder': False,
                'eval_metric': 'logloss',
            }
            if self.use_gpu:
                default_params.update({
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'gpu_id': 0
                })
            return XGBClassifier(**default_params, **self.model_params)

        elif self.model_type == 'lightgbm':
            default_params = {}
            if self.use_gpu:
                default_params.update({
                    'device': 'gpu'
                })
            return LGBMClassifier(**default_params, **self.model_params)

        elif self.model_type == 'catboost':
            default_params = {
                'verbose': 0
            }
            if self.use_gpu:
                default_params.update({
                    'task_type': 'GPU',
                    'devices': '0'
                })
            return CatBoostClassifier(**default_params, **self.model_params)

        else:
            raise ValueError("Invalid model_type. Choose from 'xgboost', 'lightgbm', or 'catboost'.")

    def fit(self, x_train:pd.DataFrame, y_train:pd.DataFrame, x_test:pd.DataFrame=None, y_test:pd.DataFrame=None):
        """
        Train the selected model on the provided dataframe.

        Parameters:
        - df (pd.DataFrame): DataFrame containing features and target
        - target_column (str): Name of the target column in the dataframe

        Returns:
        - trained model
        """

        self.model = self._get_model()
        if y_test is not None:
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Validation Accuracy ({self.model_type}, GPU={self.use_gpu}): {acc:.3f}")
        else:
            self.model.fit(x_train, y_train)

        return self.model
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() before saving.")
        
        if self.model_type == 'xgboost':
            self.model.save_model(filepath)  # saves JSON or binary .bin model
        elif self.model_type == 'lightgbm':
            self.model.booster_.save_model(filepath)
        elif self.model_type == 'catboost':
            self.model.save_model(filepath)
        else:
            raise ValueError("Unsupported model type.")

    def load_model(self, filepath):
        self.model = self._get_model()  # Create an untrained model

        if self.model_type == 'xgboost':
            self.model.load_model(filepath)
        elif self.model_type == 'lightgbm':
            booster = lgb.Booster(model_file=filepath)
            self.model._Booster = booster
        elif self.model_type == 'catboost':
            self.model.load_model(filepath)
        else:
            raise ValueError("Unsupported model type.")
        
    def predict(self, df):
        if self.model is None:
                raise ValueError("No model available. Call fit() or load_model() first.")
        return self.model.predict_proba(df)[:, 1]
