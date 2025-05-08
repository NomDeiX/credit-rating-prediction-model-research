import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib
from DataPrep import MacroDataMerger, WRDSFundamentalsDataMerger, CreditRatingDataMerger
from imblearn.over_sampling import SMOTE


class DataProcessor:
    """
    Class for loading, preprocessing and transforming raw data into features
    for credit rating forecasting model.
    """
    
    def __init__(self, data_config: Dict[str, str]):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """

        self.credit_rating_encoder = LabelEncoder()
        self.fundamentals_scaler = StandardScaler()
        self.macro_scaler = StandardScaler()
        self.fundamentals_data_path = data_config.get('fundamentals_data_path')
        self.macro_data_path = data_config.get('macro_data_path')
        self.credit_ratings_path = data_config.get('credit_ratings_path')
        
    
    def preprocess_credit_ratings(self) -> pd.DataFrame:
        """
        Preprocess credit ratings data.
        
        Args:
            df: Credit ratings DataFrame
            
        Returns:
            Preprocessed credit ratings DataFrame
        """
        creditRatingMerger = CreditRatingDataMerger(self.credit_ratings_path)
        df = creditRatingMerger.get_df()
        trend_df = df.copy()
        for col in trend_df.columns[1:]:
            na_mask = pd.isna(df[col])
            
            # Calculate diff for trend direction
            trend_df[col] = trend_df[col].diff().apply(
                lambda x: 2 if x > 0 else 0 if x < 0 else 1 if pd.notna(x) else np.nan
            )
                        
            trend_df.loc[na_mask, col] = np.nan            
            trend_df.loc[0, col] = np.nan # First row will always have NaN diff, set to NaN

        return trend_df
    
    def preprocess_fundamentals(self) -> Dict[str, pd.DataFrame]:
        """
        Preprocess fundamentals data.
        
        Args:
            df: Fundamentals DataFrame
            
        Returns:
            dict: Dictionary of fundamentals DataFrame
        """
        
        wrds = WRDSFundamentalsDataMerger(self.fundamentals_data_path)
        tickers = wrds.get_tickers_from_file(self.fundamentals_data_path)
        dict_of_fundamentals_dfs = {}

        for ticker in tickers:
            company_data = wrds.get_company_data(ticker)
            company_data['ticker'] = ticker
            company_data['debt_to_assets'] = (company_data['ltq'] / company_data['atq']).replace([float('inf'), -float('inf')], np.nan) # Debt-to-Assets ratio
            company_data['return_on_assets'] = (company_data['niq'] / company_data['atq']).replace([float('inf'), -float('inf')], np.nan) # Return on Assets (ROA)
            company_data['profit_margin'] = (company_data['niq'] / company_data['revtq']).replace([float('inf'), -float('inf')], np.nan) # Profit Margin
            company_data['interest_coverage'] = (company_data['ibq'] / company_data['xintq']).replace([float('inf'), -float('inf')], np.nan) # EBIT / Interest Expense
            dict_of_fundamentals_dfs[ticker] = company_data

        return dict_of_fundamentals_dfs
    
    def preprocess_macro(self) -> pd.DataFrame:
        """
        Preprocess macroeconomic data.
        
        Args:
            df: Macroeconomic DataFrame
            
        Returns:
            Preprocessed macroeconomic DataFrame
        """

        merger = MacroDataMerger(self.macro_data_path, start_date="1997-10-01")
        merged_macro_data = merger.merge_all()

        # Change all columns to pct change
        for col in merged_macro_data.columns[1:]:
            merged_macro_data[col] = merged_macro_data[col].pct_change()
        
        return merged_macro_data
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merge all data sources into a single DataFrame.
        
        Args:
            credit_ratings: Credit ratings DataFrame
            fundamentals: Fundamentals DataFrame
            macro: Macroeconomic DataFrame
            
        Returns:
            Merged DataFrame
        """

        ratings_df = self.preprocess_credit_ratings()
        fundamentals_dict = self.preprocess_fundamentals()
        macro_df = self.preprocess_macro()

        
        # Clean column names to keep ticker from 'exchange:ticker' format
        ratings_df.columns = [
            col if col.lower() == 'date' else col.split(':')[-1] if ':' in col else col
            for col in ratings_df.columns
        ]
        
        print(ratings_df[['Date','AME']])
        dataframes = {}
        for ticker in fundamentals_dict.keys():
            # Copy the fundamentals data for the ticker
            dataframes[ticker] = fundamentals_dict[ticker].copy()
            dataframes[ticker]['Date'] = dataframes[ticker].index

            # Align fundamentals with ratings_df
            if ticker in ratings_df.columns:
                #print(f"Processing {ticker}...")	
                shifted_ratings = ratings_df[['Date', ticker]].copy()
                shifted_ratings['Date'] = shifted_ratings['Date'] - pd.DateOffset(months=3)
                
                # Merge with shifted dates
                merged_df = pd.merge_asof(
                    dataframes[ticker],
                    shifted_ratings.rename(columns={ticker: 'rating'}),
                    on='Date',
                    direction='forward'  # Match the next shifted rating date
                )
                
                merged_df = merged_df.set_index('Date')
                dataframes[ticker] = merged_df

                

                # Merge with macro data
                merged_df = pd.merge_asof(
                    merged_df.sort_index(),
                    macro_df.sort_index(),
                    left_on='Date',
                    right_on='Date',
                    direction='forward',
                    tolerance=pd.Timedelta('31D')
                )
                merged_df = merged_df.set_index('Date')
                dataframes[ticker] = merged_df

                # Drop rows where all specified columns except 'rating' are empty
                # dataframes[ticker] = dataframes[ticker].dropna(
                #     subset=[
                #     'atq', 'ltq', 'revtq', 'dlttq', 'niq'
                # ], how='all'
                # )
        print(dataframes['AAPL'])
        return dataframes
    
    @staticmethod
    def remove_empty_data(panel: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with all NaN values in specified columns.
        
        Args:
            panel: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # remove rows where rating is NaN
        panel = panel.dropna(subset=['rating'])

        # remove rows where specified columns are all NaN
        panel = panel.dropna(subset=[
            'atq', 'ltq', 'revtq', 'ceqq', 'niq'
        ], how='all')

        # remove rows where theres more than 3 NaN values in the entire row
        num_of_columns = len(panel.columns) - 3  # Exclude 'Date', 'ticker', and 'rating'
        panel = panel.dropna(thresh=int(num_of_columns * 0.7))

        panel = panel.replace({pd.NA: np.nan})
        panel = panel.astype({c: 'float32' for c in panel.columns if c not in ['Date','ticker']})
        return panel
    
    def create_lagged_features(self, dataframe_dict: dict) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        
        Args:
            df: Merged DataFrame
            lag_periods: List of lag periods (e.g., [1, 2, 3, 4] for four quarters back)
            
        Returns:
            DataFrame with lagged features
        """
        # build a list of company frames, each with an explicit ticker column
        panel_frames = []
        for tkr, df in dataframe_dict.items():
            df = df.copy()
            df = df.reset_index()       # Date back to a column so groupby works later
            panel_frames.append(df)

        panel = pd.concat(panel_frames, ignore_index=True)
        panel = panel.sort_values(['Date', 'ticker'])

        #lags: one quarter back for every numeric column --
        lag_cols = [c for c in panel.columns if c not in ['Date', 'ticker', 'rating']]
        for lag in [1,2,3,4]:                       # add more lags later [1,2,3,4]
            panel[[f'{c}_lag{lag}' for c in lag_cols]] = (
                panel.groupby('ticker')[lag_cols]
                    .shift(lag))

        print(panel.shape)

        panel = DataProcessor.remove_empty_data(panel)

        print(panel.shape)
        print(panel)
        return panel
    
    def split_data(self, panel: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2, undersample: bool = False, smote: bool = False) -> Tuple:
        """
        Split data into training, validation, and test sets.
        
        Args:
            panel: Feature DataFrame
            test_size: Proportion for test set, if None uses fixed dates
            val_size: Proportion of remaining data for validation, if None uses fixed dates
            undersample: Whether to balance classes by undersampling the majority class
                
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Identify feature columns
        feature_cols = [c for c in panel.columns if c not in ['Date', 'ticker', 'rating']]

        # Split data either by date or by proportion
        if test_size is None or val_size is None:
            train = panel[panel['Date'] < '2023-01-01']
            valid = panel[(panel['Date'] >= '2023-01-01') & (panel['Date'] < '2024-01-01')]
            test = panel[panel['Date'] >= '2024-01-01']
        
        else:
            panel = panel.sort_values('Date')
            
            train_end_idx = int(len(panel) * (1 - test_size - val_size))
            val_end_idx = int(len(panel) * (1 - test_size))
            
            train = panel.iloc[:train_end_idx]
            valid = panel.iloc[train_end_idx:val_end_idx]
            test = panel.iloc[val_end_idx:]

         # Function to balance a dataset using undersampling
        def balance_by_undersampling(df, feature_columns):
            ratings = df['rating'].values
            unique_ratings, rating_counts = np.unique(ratings, return_counts=True)
            min_class_count = np.min(rating_counts)
            
            print(f"Original dataset: {len(df)} samples")
            print("Class distribution before undersampling:")
            for rating, count in zip(unique_ratings, rating_counts):
                print(f"  Class {rating}: {count} samples")
            
            balanced_indices = []
            for rating in unique_ratings:
                rating_indices = np.where(ratings == rating)[0]
                if len(rating_indices) > min_class_count:
                    np.random.seed(42)  # for reproducibility
                    selected_indices = np.random.choice(rating_indices, min_class_count, replace=False)
                    balanced_indices.extend(selected_indices)
                else:
                    balanced_indices.extend(rating_indices)
            
            balanced_df = df.iloc[balanced_indices].copy()
            
            balanced_ratings = balanced_df['rating'].values
            unique_ratings, rating_counts = np.unique(balanced_ratings, return_counts=True)
            print(f"Balanced dataset: {len(balanced_df)} samples")
            print("Class distribution after undersampling:")
            for rating, count in zip(unique_ratings, rating_counts):
                print(f"  Class {rating}: {count} samples")
            
            return balanced_df[feature_columns], balanced_df['rating'].values
        
        def balance_by_smote_with_nan_handling(df, feature_columns):
            X = df[feature_columns]
            y = df['rating'].values
            
            unique_ratings, rating_counts = np.unique(y, return_counts=True)
            print(f"Original dataset: {len(X)} samples")
            print("Class distribution before SMOTE:")
            for rating, count in zip(unique_ratings, rating_counts):
                print(f"  Class {rating}: {count} samples")
            

            X_original = X.copy()            
            nan_mask = X.isna() # Create a mask of missing values to restore later
            imputer = SimpleImputer(strategy='mean') # Impute missing values for SMOTE (only for creating synthetic samples)
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            # Apply SMOTE on the imputed data
            print("Applying SMOTE with imputed values for synthetic sample generation...")
            sm = SMOTE(random_state=42)
            X_resampled, y_resampled = sm.fit_resample(X_imputed, y)
            X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            
            synthetic_mask = np.ones(len(X_resampled_df), dtype=bool)
            synthetic_mask[:len(X)] = False
            
            # For original samples, restore their original values (with NaNs)
            for i, was_original in enumerate(~synthetic_mask):
                if was_original:
                    original_idx = i  # Index in the original dataset
                    X_resampled_df.iloc[i] = X_original.iloc[original_idx]
            
            unique_ratings, rating_counts = np.unique(y_resampled, return_counts=True)
            print(f"SMOTE-balanced dataset: {len(X_resampled_df)} samples")
            print("Class distribution after SMOTE:")
            for rating, count in zip(unique_ratings, rating_counts):
                print(f"  Class {rating}: {count} samples")
            
            return X_resampled_df, y_resampled
    
    
        # Apply undersampling to balance classes if requested
        if undersample:
            X_train, y_train = balance_by_undersampling(train, feature_cols)
            X_val, y_val = balance_by_undersampling(valid, feature_cols)
            X_test, y_test = balance_by_undersampling(test, feature_cols)
        
        elif smote:
            print("\nBalancing training data using SMOTE...")
            X_train_temp, y_train = balance_by_smote_with_nan_handling(train, feature_cols)
            X_val_temp, y_val = balance_by_smote_with_nan_handling(valid, feature_cols)
            X_test_temp, y_test = balance_by_smote_with_nan_handling(test, feature_cols)
            X_train = pd.DataFrame(X_train_temp, columns=feature_cols)
            X_val = pd.DataFrame(X_val_temp, columns=feature_cols)
            X_test = pd.DataFrame(X_test_temp, columns=feature_cols)

        else:
            # No balancing, just extract features and targets
            X_train = train[feature_cols]
            X_val = valid[feature_cols]
            X_test = test[feature_cols]
            
            y_train = train['rating'].values
            y_val = valid['rating'].values
            y_test = test['rating'].values
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data_pipeline(self,test_size: float = 0.2, val_size: float = 0.2, undersample: bool = False, smote: bool = False) -> Tuple:
        """
        End-to-end data pipeline from loading to train/val/test split.
        
        Args:
            credit_ratings_path: Path to credit ratings data
            fundamentals_path: Path to fundamentals data
            macro_path: Path to macroeconomic data
            lag_periods: List of lag periods
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Load data by merging
        dataframes_dict = self.merge_data()
        feature_data = self.create_lagged_features(dataframes_dict)
        
        print('arguments: ', test_size, val_size, undersample)
        # Split data
        return self.split_data(feature_data, test_size, val_size, undersample, smote)
    

class CreditRatingModel:
    """
    Class for training, evaluating, and using credit rating prediction models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the model with configuration.
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        
    def build_xgboost_model(self) -> xgb.XGBClassifier:
        """
        Build an XGBoost model with the specified parameters.
        
        Returns:
            XGBoost classifier model
        """
        model = xgb.XGBClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 6),
            learning_rate=self.config.get('learning_rate', 0.1),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.8),
            objective='multi:softprob',
            random_state=42,
            use_label_encoder=False,  # Avoid warning about label encoder
            eval_metric='mlogloss'  # Specify evaluation metric to avoid warnings
        )
        
        return model
    
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Tune hyperparameters using Grid Search with Cross-Validation and display progress.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of best hyperparameters
        """
        from tqdm.auto import tqdm
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01,0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Calculate total combinations for progress tracking
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        n_combinations *= 5  # 5-fold cross-validation
        
        print(f"Grid Search will perform {n_combinations} model fits")
        
        # Create a base model first (don't use directly in GridSearchCV)
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
            eval_metric='mlogloss'
        )
        
        # Create a pipeline with the model to ensure proper handling
        pipeline = Pipeline([
            ('classifier', base_model)
        ])
        
        # Create param grid for pipeline
        pipeline_param_grid = {f'classifier__{key}': val for key, val in param_grid.items()}
        
        # Create a progress bar callback
        class TqdmGridSearchCallback:
            def __init__(self, total):
                self.pbar = tqdm(total=total, desc="Hyperparameter Search", 
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
                self.n_completed = 0
            
            def __call__(self, params, score, n_iter):
                self.n_completed += 1
                self.pbar.update(1)
                self.pbar.set_postfix({'best_score': f"{score:.4f}"})
                return None
        
        callback = TqdmGridSearchCallback(n_combinations)
        
        # Create grid search with the callback
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=pipeline_param_grid,
            n_iter=300,  # Try 100 combinations instead of all 1215
            cv=5,
            scoring='accuracy',
            n_jobs=4,
            verbose=2,
            random_state=42
        )
        # Create grid search with the callback    
        # Fit grid search
        random_search.fit(X_train, y_train)
        
        # Close the progress bar
        callback.pbar.close()
        
        # Extract best parameters (removing 'classifier__' prefix)
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        
        # Update config with best parameters
        self.config.update(best_params)
        
        return best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> xgb.Booster:
        # Convert data to DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Set parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'max_depth': self.config.get('max_depth', 9),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.7),
            'colsample_bytree': self.config.get('colsample_bytree', 0.7),
            'eval_metric': 'mlogloss'
        }

        # Train with early stopping
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.get('n_estimators', 200),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=10
        )
        self.classes_ = np.unique(y_train)  # Store unique classes manually
        return self.model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert test data to DMatrix
        dtest = xgb.DMatrix(X_test)
        
        # Get predicted probabilities
        y_proba = self.model.predict(dtest)
        y_pred = np.argmax(y_proba, axis=1)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
            
        # Print results
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=self.classes_,  # Use stored classes
               yticklabels=self.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted classes
        """
        dmatrix = xgb.DMatrix(X)
        y_proba = self.model.predict(dmatrix)
        return np.argmax(y_proba, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability predictions
        """
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)


class ModelManager:
    """
    Class for managing the entire model lifecycle.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_processor = DataProcessor(config.get('data_config', {}))
        self.model = CreditRatingModel(config.get('model_config', {}))
        
    def train_pipeline(self) -> Dict:
        """
        End-to-end training pipeline with progress monitoring.
        
        Returns:
            Dictionary with trained model and evaluation metrics
        """
        from tqdm.auto import tqdm
        
        # Create overall progress
        stages = ['Data Preparation', 'Hyperparameter Tuning', 'Model Training', 'Model Evaluation']
        overall_progress = tqdm(total=len(stages), desc="Overall Progress", 
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        print(f"Starting stage: {stages[0]}")
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.prepare_data_pipeline(
            test_size=self.config.get('test_size', 0.2),
            val_size=self.config.get('val_size', 0.2),
            undersample=self.config.get('undersample', False),
            smote=self.config.get('smote', False)
        )
        overall_progress.update(1)
        
        # Tune hyperparameters if specified
        if self.config.get('tune_hyperparameters', False):
            print(f"\nStarting stage: {stages[1]}")
            best_params = self.model.tune_hyperparameters(X_train, y_train)
            print(f"Best hyperparameters: {best_params}")
        overall_progress.update(1)
        
        # Train model
        print(f"\nStarting stage: {stages[2]}")
        trained_model = self.model.train(X_train, y_train, X_val, y_val)
        overall_progress.update(1)
        
        # Evaluate model
        print(f"\nStarting stage: {stages[3]}")
        metrics = self.model.evaluate(X_test, y_test)
        overall_progress.update(1)
        
        overall_progress.close()
        print("\nTraining pipeline complete!")
        
        # Return results
        return {
            'model': trained_model,
            'metrics': metrics,
            'data_splits': {
                'X_train': X_train,
                'X_val': X_val, 
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
        }
    

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'data_config': {
            'macro_data_path': 'data/Macro_data.xlsx',
            'fundamentals_data_path': 'data/WRDS_features_spx.xlsx',
            'credit_ratings_path': 'data/spx_historical_credit_rating_sp_capital_iq_pro.xlsx'
        },
        'model_config': {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.01,
            'subsample': 0.6,
            'colsample_bytree': 0.7
        },
        'test_size': None, 
        'val_size': None,
        'undersample': True,
        'smote': False,
        'tune_hyperparameters': False
    }
    
    # Initialize manager
    manager = ModelManager(config)
    
    #Train pipeline
    results = manager.train_pipeline()
    print(results['metrics'])
