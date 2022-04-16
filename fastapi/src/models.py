import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import lightgbm as lgb


class do_nothing_transform(TransformerMixin, BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class lgbWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 data,
                 model_name='LGBMClassifier',
                 categorical_features=[],
                 numerical_features=[],
                 boolean_features=[],
                 target='default',
                ):
        self.model_name = model_name 
        self.data = data
        self.target = target
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.boolean_features = boolean_features
        self.model = lgb.LGBMClassifier()

    def _build_pipeline_categorical(self):
        categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        numerical_transformer = do_nothing_transform()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )
        clf = Pipeline(
            steps=[("preprocessor", preprocessor), 
                   ("model", self.model)]
        )
        return clf
    
    def _build_pipeline_numerical(self):
        
        clf = Pipeline(steps=[("model", lgb.LGBMClassifier())])
        return clf
    
    def preprocess_merchant_group(self, df):
        df['food_beverage'] = (df.merchant_group=='Food & Beverage')*1
        df['intangible'] = (df.merchant_group=='Intangible products')*1
        self.categorical_features.remove('merchant_group')
        self.numerical_features += ['food_beverage', 'intangible']
        
        return df
    
    def preprocess(self, preprocess_merchant_group):
        df_copy = self.data.copy()
        if len(self.boolean_features)>0:
            df_copy[self.boolean_features] = df_copy[self.boolean_features]*1
            self.numerical_features += self.boolean_features
        if len(self.categorical_features)>0:
            df_copy[self.categorical_features] = df_copy[self.categorical_features].fillna('NaN')
        df_copy = df_copy.fillna(-1)
        
        if preprocess_merchant_group:
            df_copy = self.preprocess_merchant_group(df_copy)
            
        return df_copy
        
    def _split_data(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df[self.categorical_features+self.numerical_features], 
                                                            df[self.target],
                                                            stratify=df[self.target], 
                                                            test_size=0.2,
                                                            random_state=42)
        self.data_split = {'train': {'X': X_train, 'y': y_train},
                           'test': {'X': X_test, 'y': y_test}}
        

    def fit(self, preprocess_merchant_group):
        
        df_final = self.preprocess(preprocess_merchant_group)
        self._split_data(df_final)
        
        if len(self.categorical_features)>0:
            clf = self._build_pipeline_categorical()
        else:
            clf = self._build_pipeline_numerical()
            
        self.opt = BayesSearchCV(
            clf,
            {
                'model__n_estimators': Integer(100, 2000),
                'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'model__reg_alpha': Real(1e-8, 10.0, prior='log-uniform'),
                'model__reg_lambda': Real(1e-8, 10.0, prior='log-uniform'),
                'model__max_depth': Integer(1, 10),
                'model__colsample_bytree': Real(0.4, 1.0),
                'model__subsample': Real(0.4, 1.0),
                'model__subsample_freq': Integer(1, 7),
                'model__min_child_samples': Integer(5, 100),
            },
            n_iter=10,
            random_state=42,
            scoring='roc_auc',
            cv=3,
            verbose=1
        )
        self.opt.fit(self.data_split['train']['X'], self.data_split['train']['y'],
            model__categorical_feature=self.categorical_features if len(self.categorical_features)>0 else 'auto',
            model__feature_name=self.categorical_features+self.numerical_features if len(self.categorical_features)>0 else 'auto'
           )

    def predict_proba(self, X):
        return self.opt.predict_proba(X)
    
    def score(self, X, y):
        return self.opt.score(X, y)