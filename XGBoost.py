import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import joblib
from DataPrep import MacroDataMerger, WRDSFundamentalsDataMerger, CreditRatingDataMerger


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
                print(f"Processing {ticker}...")	
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
    
    def split_data(self, panel: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Split data into training, validation, and test sets.
        
        Args:
            panel: Feature DataFrame
            test_size: Proportion for test set, if None uses fixed dates
            val_size: Proportion of remaining data for validation, if None uses fixed dates
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Identify feature columns
        feature_cols = [c for c in panel.columns if c not in ['Date', 'ticker', 'rating']]
        
        # Split data either by date or by proportion
        if test_size is None or val_size is None:
            # Split by fixed dates
            train = panel[panel['Date'] < '2023-01-01']
            valid = panel[(panel['Date'] >= '2023-01-01') & (panel['Date'] < '2024-01-01')]
            test = panel[panel['Date'] >= '2024-01-01']
            
            # Extract features and target for each set
            X_train = train[feature_cols]
            X_val = valid[feature_cols]
            X_test = test[feature_cols]
            
            # Extract target values
            y_train = train['rating'].values
            y_val = valid['rating'].values
            y_test = test['rating'].values
        else:
            # Split by proportion
            # Sort by date first to ensure time-based split
            panel = panel.sort_values('Date')
            
            # Determine split points
            train_end_idx = int(len(panel) * (1 - test_size - val_size))
            val_end_idx = int(len(panel) * (1 - test_size))
            
            # Split the dataframe
            train_df = panel.iloc[:train_end_idx]
            val_df = panel.iloc[train_end_idx:val_end_idx]
            test_df = panel.iloc[val_end_idx:]
            
            # Extract features for each set
            X_train = train_df[feature_cols]
            X_val = val_df[feature_cols]
            X_test = test_df[feature_cols]
            
            # Extract target values
            y_train = train_df['rating'].values
            y_val = val_df['rating'].values
            y_test = test_df['rating'].values
        
        # Scale features (same for both splitting methods)
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data_pipeline(self,
                            test_size: float = 0.2,
                            val_size: float = 0.2) -> Tuple:
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
        
        # Split data
        return self.split_data(feature_data)
    

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
            Tune hyperparameters using Grid Search with Cross-Validation.
            
            Args:
                X_train: Training features
                y_train: Training target
                
            Returns:
                Dictionary of best hyperparameters
            """
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
            # Create a base model first (don't use directly in GridSearchCV)
            base_model = xgb.XGBClassifier(
                objective='multi:softprob',
                random_state=42,
                use_label_encoder=False,  # Avoid warning about label encoder
                verbosity=0,  # Reduce verbosity to avoid warnings
                eval_metric='mlogloss'  # Specify evaluation metric to avoid warnings
            )
            
            # Create a pipeline with the model to ensure proper handling
            pipeline = Pipeline([
                ('classifier', base_model)
            ])
            
            # Create param grid for pipeline
            pipeline_param_grid = {f'classifier__{key}': val for key, val in param_grid.items()}
            
            # Create grid search
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=pipeline_param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
        
            # Extract best parameters (removing 'classifier__' prefix)
            best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
            
            # Update config with best parameters
            self.config.update(best_params)
            
            return best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
        """
        Train the model on the training data and validate.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained model
        """
        # Build the model
        self.model = self.build_xgboost_model()
        
        # Train the model
        self.model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=['mlogloss', 'merror'],
            early_stopping_rounds=20,
            verbose=True
        )
        
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
        # Make predictions
        y_pred = self.model.predict(X_test)
        
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
                   xticklabels=self.model.classes_, 
                   yticklabels=self.model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        return metrics
    
    def feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame of feature importances
        """
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, path)
        
        # Save configuration
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        pd.Series(self.config).to_json(config_path)
        
        print(f"Model saved to {path}")
        print(f"Configuration saved to {config_path}")
    
    def load_model(self, path: str) -> xgb.XGBClassifier:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Load the model
        self.model = joblib.load(path)
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        if os.path.exists(config_path):
            self.config = pd.read_json(config_path, typ='series').to_dict()
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted classes
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability predictions
        """
        return self.model.predict_proba(X)


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
        End-to-end training pipeline.
        
        Args:
            credit_ratings_path: Path to credit ratings data
            fundamentals_path: Path to fundamentals data
            macro_path: Path to macroeconomic data
            
        Returns:
            Dictionary with trained model and evaluation metrics
        """
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.prepare_data_pipeline(
            test_size=self.config.get('test_size', 0.2),
            val_size=self.config.get('val_size', 0.2)
        )
        
        # Tune hyperparameters if specified
        if self.config.get('tune_hyperparameters', False):
            best_params = self.model.tune_hyperparameters(X_train, y_train)
            print(f"Best hyperparameters: {best_params}")
        
        # Train model
        trained_model = self.model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = self.model.evaluate(X_test, y_test)
        
        # Get feature importance
        feature_importance = self.model.feature_importance(X_train.columns.tolist())
        
        # Save model if specified
        if self.config.get('save_model', False):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(
                self.config.get('model_save_dir', './models'),
                f"credit_rating_model_{timestamp}.pkl"
            )
            self.model.save_model(model_path)
        
        # Return results
        return {
            'model': trained_model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'data_splits': {
                'X_train': X_train,
                'X_val': X_val, 
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
        }
    
    def predict_pipeline(self, 
                       model_path: str,
                       new_data_path: str) -> pd.DataFrame:
        """
        End-to-end prediction pipeline.
        
        Args:
            model_path: Path to the saved model
            new_data_path: Path to new data for prediction
            
        Returns:
            DataFrame with predictions
        """
        # Load model
        self.model.load_model(model_path)
        
        # Load and preprocess new data
        # This would depend on the format of the new data
        # For simplicity, assuming it's already in the right format
        new_data = pd.read_csv(new_data_path)
        
        # Make predictions
        predictions = self.model.predict(new_data)
        probabilities = self.model.predict_proba(new_data)
        
        # Add predictions to data
        results = new_data.copy()
        results['predicted_class'] = predictions
        
        for i, cls in enumerate(self.model.model.classes_):
            results[f'probability_{cls}'] = probabilities[:, i]
        
        return results


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
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'test_size': 0.2,
        'val_size': 0.2,
        'tune_hyperparameters': True,
        'save_model': False,
        'model_save_dir': './models'
    }
    
    # Initialize manager
    manager = ModelManager(config)
    
    #Train pipeline
    results = manager.train_pipeline()
    print(results['metrics'])
