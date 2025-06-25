import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from typing import Type, List, Callable, Optional

class ModelSelector:
    def __init__(self, model_type: str = 'xgboost', use_gpu: bool = False, test_size: float = 0.2, random_state: int = 42, **model_params)->None:
        """Initialize the ModelSelector with a specified model.
        Parameters
        -----------------------------------
        model_type: str
            The framework to use. Possible options are ['xgboost', 'lightgbm', 'catboost']
        use_gpu: bool
            Whether to enable GPU acceleration
        test_size: float 
            Fraction of data to use for validation
        random_state: int
            Seed for reproducibility 
        model_params: dict
            Additional keyword arguments to pass to the model
        Returns
        -----------------------------------
        """
        self.model_type = model_type.lower()
        self.use_gpu = use_gpu
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params
        self.model = None

    def _get_model(self)->Type[XGBClassifier|LGBMClassifier|CatBoostClassifier]:
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

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame = None, y_test: pd.DataFrame = None):
        """
        Train the selected model on the provided dataframe.
        Parameters
        -----------------------------------
        x_train: pd.DataFrame
            The pandas dataframe for training features
        y_train: pd.Series
            The pandas Series for training lables
        x_test: pd.DataFrame
            The pandas dataframe for validation features
        y_test: pd.Series
            The pandas series for validation labels
        Returns
        -----------------------------------
        Type[XGBClassifier|LGBMClassifier|CatBoostClassifier]
            Returns trained model
        """
        self.model = self._get_model()
        if y_test is not None:
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            print(f"Validation Accuracy ({self.model_type}, GPU={self.use_gpu}): {acc:.3f}")
        else:
            self.model.fit(x_train, y_train)

        return self.model
    
    def save_model(self, filepath: str)->None:
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

    def load_model(self, filepath: str)->None:
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
        
    def predict(self, df: pd.DataFrame)->np.array:
        if self.model is None:
                raise ValueError("No model available. Call fit() or load_model() first.")
        return self.model.predict_proba(df)[:, 1]

def get_scorer(metric: str = 'f1'):
    """Returns function to compute the score asked for
    """
    metric = metric.lower()
    classification_metrics = {
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "f1": metrics.f1_score,
        "roc_auc": metrics.roc_auc_score,
        "balanced_accuracy": metrics.balanced_accuracy_score
    }
    if metric not in list(classification_metrics.keys()):
        raise ValueError(f"The supported metrics are {list(classification_metrics.keys())}")

    return classification_metrics[metric]

def optuna_objective(trial: Type[optuna.trial.Trial], x_train: pd.DataFrame , y_train: pd.Series, sample_weights: np.ndarray,
                     x_validation: pd.DataFrame, y_validation: pd.Series, sample_weights_val: Optional[np.ndarray] = None, 
                     metric: str ='f1', enable_categorical: bool = False)->float:
    """Returns an objective for optuna to extremize. 
    Parameters
    -----------------------------------
    trial: Type[optuna.trial.Trial]
        An optuna trial (process for evaluating objective function).
    x_train: pd.DataFrame
        Pandas dataframe for the training dataset
    y_train: pd.Series
        Pandas series for training labels
    sample_weights: np.ndarray
        Numpy array for sample weights for train set. Result of sklearn `compute_sample_weights` function. 
    x_validation: pd.DataFrame
        Pandas dataframe for the validation dataset
    y_validation: pd.Series
        Pandas series for validation labels
    sample_weights_val: np.ndarray
        Numpy array for sample weights for validaton Set. Optional
    metric: str
        The metric to use for optimization. Supported metrics are: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy'] 
    enable_categorical: bool
        Whether to enable categorical features handling, an experimental feature of XGBoost
        
    Returns
    -----------------------------------
    float
        The score corresponding to selected metric.
    """
    scorer = get_scorer(metric)
    params = {
       'tree_method': 'hist',
       'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 0.001, 25, log=True),
        'reg_alpha': trial.suggest_float('alpha', 0, 25),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.8),
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc', 'aucpr'],
        'n_estimators': trial.suggest_int('n_estimators', 40, 100),
        'max_delta_step': trial.suggest_float('max_delta_step', 1, 9),
        'early_stopping_rounds': 5
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'validation_1-logloss')
    bdt_model = XGBClassifier(**params, callbacks=[pruning_callback], enable_categorical = enable_categorical)
    bdt_model.fit(x_train, y_train, sample_weight=sample_weights, eval_set=[(x_train, y_train), (x_validation,y_validation)], 
                  sample_weight_eval_set=[sample_weights, sample_weights_val])
    
    y_pred_val =  bdt_model.predict(x_validation)
    score = scorer(y_validation, y_pred_val)
    return score

def hyper_parameter_optimization(x_train: pd.DataFrame, y_train: pd.Series, x_validation: pd.DataFrame, y_validation: pd.Series, 
                                 sample_weights: np.ndarray|None = None, sample_weights_val: np.ndarray|None = None, 
                                 metric: str = 'f1', n_trials: int = 50, enable_categorical: bool = False)->dict:
    """A function for hyper parameter optimization using optuna library
    Parameters
    -----------------------------------
    x_train: pd.DataFrame
        Pandas dataframe for the training dataset
    y_train: pd.Series
        Pandas series for training labels
    x_validation: pd.DataFrame
        Pandas dataframe for the validation dataset
    y_validation: pd.Series
        Pandas series for validation labels
    sample_weights: np.ndarray|None
        Numpy array for sample weights for train set. Result of sklearn `compute_sample_weights` function. 
    sample_weights_val: np.ndarray|None
        Numpy array for sample weights for validaton Set. Optional
    metric: str
        The metric to use for optimization. Supported metrics are: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy'] 
    n_trials: int
        The number of trials run for optimization.
    enable_categorical: bool
        Whether to enable categorical feature handling, an experimental feature of XGBoost

    Returns
    -----------------------------------
    dict
        A dictionary with best parameters based on maximizing the given metric
    """
    sampler = optuna.samplers.TPESampler(seed=123)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: optuna_objective(trial, x_train, y_train, sample_weights, x_validation, y_validation, sample_weights_val, 
                                                  metric=metric, enable_categorical = enable_categorical), n_trials = n_trials)
    best_params = study.best_params
    return best_params

def hyper_parameter_optimization_cv(x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42, n_trials: int = 100, n_jobs: int = 1, 
                                    metric: str = 'f1', enable_categorical: bool = False)->dict:
    """A function to optimize hyper parameters using optuna library with cross validation
    Parameters
    -----------------------------------
    x_train: pd.DataFrame
        The pandas dataframe of train features
    y_train: pd.Series
        The pandas series of train labels
    random_state: int
        The seed for reproducibility
    n_trials: int
        The number of trials to run.
    n_jobs: int
        Number of jobs to run concurrently. Set to -1 for using all processors
    metric: str
        The metric to use for optimization. Supported metrics are: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy']
    enable_categorical: bool
        Whether to enable categorical feature handling, an experimental feature of XGBoost

    Returns
    -----------------------------------
    dict
        A dictionary with best hyper-parameters
    """
    clf = XGBClassifier(enable_categorical = enable_categorical, tree_method="hist")
    params = {
        'max_depth': optuna.distributions.IntDistribution(3, 12),
        'subsample': optuna.distributions.FloatDistribution(0.6, 1.0),
        'colsample_bynode': optuna.distributions.FloatDistribution(0.6, 1.0),
        'lambda': optuna.distributions.FloatDistribution(0.001, 25, log=True),
        'reg_alpha': optuna.distributions.FloatDistribution(0, 25),
        'learning_rate': optuna.distributions.FloatDistribution(0.001, 0.8),
        'n_estimators': optuna.distributions.IntDistribution(40, 1000),
        'max_delta_step': optuna.distributions.FloatDistribution(1, 9),
    }

    optuna_search = optuna.integration.OptunaSearchCV(clf, params, random_state = random_state, n_trials = n_trials, 
                                                      scoring = metrics.make_scorer(get_scorer(metric)),
                                                      n_jobs = n_jobs)
    optuna_search.fit(x_train, y_train)
    return optuna_search.best_params_
