import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class UnifiedSegmentedModel:
    """
    A unified model that combines segment-specific models with a meta-learner
    for insurance settlement prediction. This model internally segments the data
    and trains specialized models for each segment, then combines their predictions.
    """
    
    def __init__(self, segmentation_cols=['Exceptional_Circumstances', 'AccidentType'], 
                 min_segment_size=50, use_feature_selection=True, verbose=True):
        """
        Initialize the unified segmented model.
        
        Args:
            segmentation_cols: List of columns to use for segmentation
            min_segment_size: Minimum number of samples required for a segment
            use_feature_selection: Whether to use feature selection
            verbose: Whether to print progress information
        """
        self.segmentation_cols = segmentation_cols
        self.min_segment_size = min_segment_size
        self.use_feature_selection = use_feature_selection
        self.verbose = verbose
        
        # Will be set during training
        self.segments = {}
        self.segment_models = {}
        self.meta_model = None
        self.feature_columns = None
        self.is_trained = False
        
    def _log(self, message):
        """Print message if verbose is enabled"""
        if self.verbose:
            print(message)
    
    def _preprocess_data(self, df):
        """Preprocess data with cleaning and handling missing values"""
        df = df.copy()  # Make a copy to avoid modifying the original
        
        # Handle missing values in the target variable if present
        if 'SettlementValue' in df.columns and df['SettlementValue'].isna().any():
            self._log("Handling missing values in target variable")
            if self.is_trained:  # For prediction, we don't want to drop rows
                # Just log the info but keep rows
                self._log(f"Note: {df['SettlementValue'].isna().sum()} rows have missing target values")
            else:  # During training, we need to drop rows with missing targets
                df = df.dropna(subset=['SettlementValue'])
        
        # Get numeric columns (excluding target if present)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if 'SettlementValue' in numeric_cols:
            numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        # Handle missing values in numeric columns with median
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing values in categorical columns with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _engineer_features(self, df):
        """Add engineered features to improve model performance"""
        self._log("Engineering features")
        df_new = df.copy()
        
        # Time between accident and claim filing
        if 'Accident Date' in df.columns and 'Claim Date' in df.columns:
            try:
                df_new['Days_To_Claim'] = (pd.to_datetime(df_new['Claim Date']) - 
                                          pd.to_datetime(df_new['Accident Date'])).dt.days
            except:
                pass
        
        # Total special damages (sum of all Special* columns)
        special_columns = [col for col in df.columns if col.startswith('Special')]
        if len(special_columns) > 0:
            df_new['Total_Special_Damages'] = df_new[special_columns].sum(axis=1)
            df_new['Log_Special_Damages'] = np.log1p(df_new['Total_Special_Damages'])
        
        # Total general damages
        general_columns = [col for col in df.columns if col.startswith('General')]
        if len(general_columns) > 0:
            df_new['Total_General_Damages'] = df_new[general_columns].sum(axis=1)
            df_new['Log_General_Damages'] = np.log1p(df_new['Total_General_Damages'])
        
        # Total damages
        if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
            df_new['Total_Damages'] = df_new['Total_Special_Damages'] + df_new['Total_General_Damages']
            df_new['Log_Total_Damages'] = np.log1p(df_new['Total_Damages'])
        
        # Ratio features
        if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
            # Avoid division by zero
            df_new['Special_General_Ratio'] = df_new['Total_Special_Damages'] / df_new['Total_General_Damages'].replace(0, 1)
        
        # Create injury severity flags
        for col in df_new.select_dtypes(include=['object']).columns:
            if 'Injury' in col or 'injury' in col:
                df_new['Severe_Injury_Flag'] = df_new[col].str.lower().str.contains('severe|major|critical', na=False).astype(int)
                break
        
        # Whiplash specific flag (common in car accidents)
        if 'Whiplash' in df_new.columns:
            df_new['Whiplash_Numeric'] = (df_new['Whiplash'] == 'Yes').astype(int)
        
        # Psychological injury impacts
        if 'Minor_Psychological_Injury' in df_new.columns:
            df_new['Psych_Injury_Numeric'] = (df_new['Minor_Psychological_Injury'] == 'Yes').astype(int)
        
        # Create complex interaction features
        if 'Severe_Injury_Flag' in df_new.columns and 'Total_Damages' in df_new.columns:
            df_new['Severe_Injury_Damages'] = df_new['Severe_Injury_Flag'] * df_new['Total_Damages']
        
        if 'Psych_Injury_Numeric' in df_new.columns and 'Total_Damages' in df_new.columns:
            df_new['Psych_Damage_Interaction'] = df_new['Psych_Injury_Numeric'] * df_new['Total_Damages']
        
        return df_new
    
    def _create_segments(self, df):
        """Create segments based on specified columns"""
        self._log(f"Creating segments based on {self.segmentation_cols}")
        segments = {}
        
        # For single column segmentation
        if len(self.segmentation_cols) == 1:
            col = self.segmentation_cols[0]
            for value in df[col].unique():
                segment_df = df[df[col] == value]
                if len(segment_df) >= self.min_segment_size:
                    segments[f"{value}"] = segment_df
        
        # For multi-column segmentation
        else:
            # Create a combined segment identifier
            df = df.copy()
            df['_segment_id'] = ''
            for col in self.segmentation_cols:
                df['_segment_id'] += df[col].astype(str) + '_'
            
            # Group by the combined segment ID
            for seg_id, segment_df in df.groupby('_segment_id'):
                if len(segment_df) >= self.min_segment_size:
                    segments[seg_id] = segment_df.drop('_segment_id', axis=1)
        
        # Create "Other" segment for records not in any segment
        all_segmented = pd.concat([segment_df for segment_df in segments.values()])
        other_records = df.loc[~df.index.isin(all_segmented.index)]
        
        if len(other_records) >= self.min_segment_size:
            segments['Other'] = other_records
        else:
            # Find largest segment to add remaining records to
            largest_segment = max(segments.items(), key=lambda x: len(x[1]))
            segments[largest_segment[0]] = pd.concat([largest_segment[1], other_records])
        
        self._log(f"Created {len(segments)} segments")
        return segments
    
    def _get_segment_for_record(self, record):
        """Determine which segment a record belongs to"""
        # For single column segmentation
        if len(self.segmentation_cols) == 1:
            col = self.segmentation_cols[0]
            value = record[col]
            segment_id = f"{value}"
            if segment_id in self.segments:
                return segment_id
        
        # For multi-column segmentation
        else:
            # Create combined segment identifier
            segment_id = ''
            for col in self.segmentation_cols:
                segment_id += str(record[col]) + '_'
            
            if segment_id in self.segments:
                return segment_id
        
        # Return 'Other' if no matching segment found
        return 'Other'
    
    def _select_features(self, X, y):
        """Select important features using XGBoost feature importance"""
        # Quick check if we have enough features to do selection
        if X.shape[1] <= 5:
            return X, X.columns.tolist()
        
        # Use XGBoost for feature selection
        feature_selector = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        feature_selector.fit(X, y)
        
        # Select features based on importance
        selection_model = SelectFromModel(
            feature_selector, 
            threshold='mean',  # Use mean importance as threshold
            prefit=True
        )
        
        # Get selected feature indices
        selected_indices = selection_model.get_support()
        
        # Get selected feature names
        selected_features = X.columns[selected_indices].tolist()
        
        # Ensure we have at least 3 features
        if len(selected_features) < 3:
            # Select top 3 by importance
            importances = feature_selector.feature_importances_
            top_indices = np.argsort(importances)[-3:]
            selected_features = X.columns[top_indices].tolist()
        
        # Return selected features
        return X[selected_features], selected_features
    
    def _train_segment_model(self, segment_df, segment_name):
        """Train a model for a specific segment"""
        self._log(f"Training model for segment: {segment_name} (n={len(segment_df)})")
        
        # Get numeric features for modeling (categorical would require encoding)
        numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        # Prepare X and y
        X = segment_df[numeric_cols].copy()
        y = segment_df['SettlementValue']
        
        # Apply feature selection if enabled
        if self.use_feature_selection and X.shape[1] > 5:
            X_selected, selected_features = self._select_features(X, y)
        else:
            X_selected = X
            selected_features = X.columns.tolist()
        
        # Train the model with reasonable default parameters
        model = XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train the model
        model.fit(X_selected, y)
        
        # Evaluate on training data to get a sense of performance
        y_pred = model.predict(X_selected)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        self._log(f"  RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        # Return trained model and metadata
        return {
            'model': model,
            'features': selected_features,
            'metrics': {
                'rmse': rmse,
                'r2': r2
            },
            'size': len(segment_df)
        }
    
    def _create_meta_features(self, df):
        """Create meta-features for stacking ensemble"""
        self._log("Creating meta-features for stacking ensemble")
        
        # Get segment assignment for each record
        df_meta = df.copy()
        
        try:
            df_meta['segment_id'] = df_meta.apply(self._get_segment_for_record, axis=1)
        except Exception as e:
            self._log(f"Error assigning segments: {str(e)}")
            # Assign all to 'Other'
            df_meta['segment_id'] = 'Other'
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
        
        # For each segment, add predictions from its model
        for segment_name, model_info in self.segment_models.items():
            # Initialize prediction column
            df_meta[f'pred_{segment_name}'] = np.nan
            
            # Get segment model and features
            segment_model = model_info['model']
            segment_features = model_info['features']
            
            # Make sure all required features exist
            missing_features = [f for f in segment_features if f not in df_meta.columns]
            if missing_features:
                self._log(f"Warning: Missing features for segment {segment_name}: {missing_features}")
                continue
            
            # Get rows for this segment
            segment_mask = df_meta['segment_id'] == segment_name
            
            if any(segment_mask):
                try:
                    # Prepare features and predict
                    X_segment = df_meta.loc[segment_mask, segment_features]
                    preds = segment_model.predict(X_segment)
                    df_meta.loc[segment_mask, f'pred_{segment_name}'] = preds
                except Exception as e:
                    self._log(f"Error predicting for segment {segment_name}: {str(e)}")
        
        # Create default meta-features that will work even if segment predictions fail
        df_meta['primary_segment_pred'] = df['SettlementValue'].mean() if 'SettlementValue' in df.columns else 0
        
        # Create a few simple meta-features from numeric columns
        if 'Total_Special_Damages' in df_meta.columns and 'Total_General_Damages' in df_meta.columns:
            df_meta['total_damages_meta'] = df_meta['Total_Special_Damages'] + df_meta['Total_General_Damages']
        
        # Ensure we have at least some meta-features to work with
        meta_features = [col for col in df_meta.columns if 'pred' in col or 'meta' in col or 
                        col in ['Total_Special_Damages', 'Total_General_Damages']]
        
        if not meta_features:
            self._log("Warning: No meta-features created, using numeric columns")
            meta_features = numeric_cols
        
        # Make sure all columns exist and have no NaN values
        final_features = [col for col in meta_features if col in df_meta.columns]
        return df_meta[final_features].fillna(0)
    
    def _build_meta_model(self, X_meta, y):
        """Build a meta-model that combines segment predictions"""
        self._log("Building meta-model")
        
        # Build a stacking ensemble
        base_models = [
            ('xgb', XGBRegressor(learning_rate=0.05, max_depth=4, n_estimators=150, random_state=42)),
            ('gbm', GradientBoostingRegressor(learning_rate=0.05, max_depth=4, n_estimators=150, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42))
        ]
        
        # Final estimator
        final_estimator = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
        
        # Create stacking regressor
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )
        
        # Train the model
        stacking_regressor.fit(X_meta, y)
        
        # Evaluate
        y_pred = stacking_regressor.predict(X_meta)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        self._log(f"Meta-model performance - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        return stacking_regressor
    
    def fit(self, df):
        """
        Train the unified segmented model on the given data.
        
        Args:
            df: DataFrame containing features and target 'SettlementValue'
        
        Returns:
            self: The trained model instance
        """
        start_time = time.time()
        self._log("Training unified segmented model")
        
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Store feature columns for prediction
        self.feature_columns = df.columns.tolist()
        
        # Create segments
        self.segments = self._create_segments(df)
        
        # Train models for each segment
        self.segment_models = {}
        for segment_name, segment_df in self.segments.items():
            self.segment_models[segment_name] = self._train_segment_model(segment_df, segment_name)
        
        # Create meta-features for stacking
        meta_features = self._create_meta_features(df)
        
        # Build meta-model
        self.meta_model = self._build_meta_model(meta_features, df['SettlementValue'])
        
        # Mark as trained
        self.is_trained = True
        
        # Log training time
        end_time = time.time()
        training_time = end_time - start_time
        self._log(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, df):
        """
        Make predictions using the trained unified model.
        
        Args:
            df: DataFrame containing features (without target)
        
        Returns:
            array: Predicted settlement values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Keep track of original indices
        original_indices = df.index.copy()
        
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Create meta-features
        meta_features = self._create_meta_features(df)
        
        # Make predictions using meta-model
        try:
            predictions = self.meta_model.predict(meta_features)
            self._log(f"Made predictions for {len(predictions)} records")
            
            # If indices changed, we need to handle this
            if len(predictions) != len(original_indices):
                self._log(f"Warning: Original data had {len(original_indices)} records, but predictions were made for {len(predictions)}")
            
            return predictions
        except Exception as e:
            self._log(f"Error during prediction: {str(e)}")
            # Return empty array
            return np.array([])
    
    def score(self, df):
        """
        Compute R² score on the given data.
        
        Args:
            df: DataFrame containing features and target 'SettlementValue'
        
        Returns:
            float: R² score
        """
        if 'SettlementValue' not in df.columns:
            raise ValueError("Target 'SettlementValue' not found in data")
        
        # Make a copy so we don't modify the original
        df_copy = df.copy()
        
        # Get predictions
        try:
            y_pred = self.predict(df_copy)
            
            # Make sure we only score on rows where we have predictions
            if len(y_pred) != len(df_copy):
                self._log(f"Warning: Predictions were made for {len(y_pred)} out of {len(df_copy)} records")
                # Keep only rows that have predictions
                df_copy = df_copy.iloc[:len(y_pred)]
            
            y_true = df_copy['SettlementValue']
            
            # Make sure both arrays have the same length
            min_len = min(len(y_true), len(y_pred))
            return r2_score(y_true[:min_len], y_pred[:min_len])
            
        except Exception as e:
            self._log(f"Error during scoring: {str(e)}")
            # Return a default value
            return float('nan')
    
    def save(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        joblib.dump(self, filepath)
        self._log(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            UnifiedSegmentedModel: Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


# Example usage:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Synthetic_Data_For_Students.csv')
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create and train the unified model
    model = UnifiedSegmentedModel(
        segmentation_cols=['Exceptional_Circumstances', 'AccidentType'],
        min_segment_size=50,
        use_feature_selection=True,
        verbose=True
    )
    
    # Train the model
    model.fit(train_df)
    
    # Evaluate on test set
    test_r2 = model.score(test_df)
    print(f"Test R² Score: {test_r2:.4f}")
    
    # Make predictions for new data
    predictions = model.predict(test_df)
    
    # Save the model
    model.save('unified_segmented_model.pkl')
    
    # Later, load the model
    loaded_model = UnifiedSegmentedModel.load('unified_segmented_model.pkl')
    
    # Use the loaded model
    new_predictions = loaded_model.predict(test_df)