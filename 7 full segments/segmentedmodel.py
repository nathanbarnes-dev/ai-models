import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('output', exist_ok=True)

# Create a log file
log_filename = f'output/enhanced_model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log_file = open(log_filename, 'w')

# Function to log to both console and file
def log(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

# Load the dataset
def load_data(file_path):
    """Load the dataset from CSV file"""
    df = pd.read_csv(file_path)
    log(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Preprocess data
def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Handle missing values in the target variable
    log("\n--- Handling Missing Values and Outliers ---")
    missing_in_target = df['SettlementValue'].isna().sum()
    log(f"Missing values in SettlementValue: {missing_in_target}")
    
    # Remove rows with missing target values
    if missing_in_target > 0:
        log(f"Removing {missing_in_target} rows with missing SettlementValue")
        df = df.dropna(subset=['SettlementValue'])
    
    # Check for infinity or extremely large values
    df = df[np.isfinite(df['SettlementValue'])]
    log(f"Shape after removing infinity values from target: {df.shape}")
    
    # Cap extremely large values at a reasonable threshold (e.g., 99.9 percentile)
    upper_threshold = df['SettlementValue'].quantile(0.999)
    df = df[df['SettlementValue'] <= upper_threshold]
    log(f"Shape after removing extreme outliers from target: {df.shape}")
    
    # Fill missing values in numeric columns with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing values in categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    return df

# Feature engineering
def engineer_features(df):
    """Add engineered features to improve model performance"""
    log("\n--- Adding Engineered Features ---")
    
    # Create a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Count of engineered features added
    feature_count = 0
    
    # Time between accident and claim filing
    if 'Accident Date' in df.columns and 'Claim Date' in df.columns:
        try:
            df_new['Days_To_Claim'] = (pd.to_datetime(df_new['Claim Date']) - 
                                      pd.to_datetime(df_new['Accident Date'])).dt.days
            feature_count += 1
        except:
            log("Could not calculate days between dates")
    
    # Total special damages (sum of all Special* columns)
    special_columns = [col for col in df.columns if col.startswith('Special')]
    if len(special_columns) > 0:
        df_new['Total_Special_Damages'] = df_new[special_columns].sum(axis=1)
        feature_count += 1
    
    # Total general damages
    general_columns = [col for col in df.columns if col.startswith('General')]
    if len(general_columns) > 0:
        df_new['Total_General_Damages'] = df_new[general_columns].sum(axis=1)
        feature_count += 1
    
    # Ratio features
    if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
        # Avoid division by zero
        df_new['Special_General_Ratio'] = df_new['Total_Special_Damages'] / df_new['Total_General_Damages'].replace(0, 1)
        feature_count += 1
    
    # Age-related damages ratio (age may affect settlement)
    if 'Driver Age' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Age_Damage_Ratio'] = df_new['Driver Age'] / df_new['Total_Special_Damages'].replace(0, 1)
        feature_count += 1
    
    # Create category-specific engineered features
    
    # Serious injury flag
    injury_flag_created = False
    for col in df_new.select_dtypes(include=['object']).columns:
        if 'Injury' in col or 'injury' in col:
            # Check if 'severe' or 'major' appears in the text
            if not injury_flag_created:
                df_new['Severe_Injury_Flag'] = df_new[col].str.lower().str.contains('severe|major|critical', na=False).astype(int)
                injury_flag_created = True
                feature_count += 1
    
    # Whiplash specific flag (common in car accidents)
    if 'Whiplash' in df_new.columns:
        df_new['Whiplash_Numeric'] = (df_new['Whiplash'] == 'Yes').astype(int)
        feature_count += 1
    
    # Create interaction features
    if 'Severe_Injury_Flag' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Severe_Injury_Damages'] = df_new['Severe_Injury_Flag'] * df_new['Total_Special_Damages']
        feature_count += 1
    
    # Vehicle age interactions
    if 'Vehicle Age' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Vehicle_Age_Damage_Interaction'] = df_new['Vehicle Age'] * df_new['Total_Special_Damages'] / 1000
        feature_count += 1
    
    # Psychological injury impacts
    if 'Minor_Psychological_Injury' in df_new.columns and 'Total_Special_Damages' in df_new.columns:
        df_new['Psych_Injury_Numeric'] = (df_new['Minor_Psychological_Injury'] == 'Yes').astype(int)
        df_new['Psych_Damage_Interaction'] = df_new['Psych_Injury_Numeric'] * df_new['Total_Special_Damages']
        feature_count += 2
    
    # Polynomial features for key numeric columns
    for col in ['Total_Special_Damages', 'Total_General_Damages']:
        if col in df_new.columns:
            df_new[f'{col}_Squared'] = df_new[col] ** 2
            feature_count += 1
    
    log(f"Added {feature_count} engineered features")
    return df_new

# Create multi-level segmentation 
def create_multi_segments(df, primary_col='Exceptional_Circumstances', secondary_col='AccidentType', min_segment_size=50):
    """Create a multi-level segmentation approach"""
    log(f"\n--- Creating Multi-level Segmentation ({primary_col} + {secondary_col}) ---")
    
    segments = {}
    
    # Get unique values for primary column
    primary_values = df[primary_col].unique()
    
    for primary_value in primary_values:
        # Get data for this primary value
        primary_df = df[df[primary_col] == primary_value]
        
        if len(primary_df) < min_segment_size:
            log(f"Primary segment '{primary_value}' has only {len(primary_df)} rows, below minimum {min_segment_size}")
            continue
        
        # Check if further segmentation is beneficial
        if len(primary_df) >= 3 * min_segment_size:
            # Get unique values for secondary column within this primary segment
            secondary_values = primary_df[secondary_col].unique()
            
            for secondary_value in secondary_values:
                # Get data for this combination
                segment_df = primary_df[primary_df[secondary_col] == secondary_value]
                
                if len(segment_df) >= min_segment_size:
                    segment_name = f"{primary_value}_{secondary_value}"
                    segments[segment_name] = segment_df
                    log(f"Created segment '{segment_name}' with {len(segment_df)} rows")
        else:
            # Just use primary segmentation
            segments[primary_value] = primary_df
            log(f"Created segment '{primary_value}' with {len(primary_df)} rows")
    
    # Create a catch-all 'Other' segment for any data not in the above segments
    all_segmented_indices = []
    for segment_df in segments.values():
        all_segmented_indices.extend(segment_df.index.tolist())
    
    other_df = df.loc[~df.index.isin(all_segmented_indices)]
    
    if len(other_df) >= min_segment_size:
        segments['Other'] = other_df
        log(f"Created 'Other' segment with {len(other_df)} rows")
    
    log(f"Created a total of {len(segments)} segments")
    return segments

# Feature selection for a segment
def select_features_for_segment(X, y, segment_name):
    """Select important features for a segment using the feature importance from XGBoost"""
    log(f"Selecting features for segment: {segment_name}")
    
    # Initialize a simple model for feature importance
    feature_selector = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Fit the model
    feature_selector.fit(X, y)
    
    # Create selector based on feature importance
    selection_model = SelectFromModel(
        feature_selector, 
        threshold='mean',  # Use the mean importance as threshold
        prefit=True
    )
    
    # Transform X to select features
    X_selected = selection_model.transform(X)
    
    # Get selected feature names
    selected_indices = selection_model.get_support()
    selected_features = X.columns[selected_indices].tolist()
    
    log(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    
    return X_selected, selected_features, selection_model

# Tune hyperparameters for a segment
def tune_hyperparameters_for_segment(X, y, segment_name):
    """Tune XGBoost hyperparameters for a specific segment"""
    log(f"Tuning hyperparameters for segment: {segment_name}")
    
    # Define parameter grid - smaller for faster testing
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Create model
    xgb_model = XGBRegressor(random_state=42)
    
    # Create K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=kfold,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit GridSearchCV
    grid_search.fit(X, y)
    
    # Get best parameters
    best_params = grid_search.best_params_
    log(f"Best hyperparameters: {best_params}")
    
    # Create model with best parameters
    best_model = XGBRegressor(
        random_state=42,
        **best_params
    )
    
    return best_model, best_params

# Build model for a segment
def build_model_for_segment(segment_df, segment_name, tune_hyperparams=True, select_features=True):
    """Build a model for a single segment with feature selection and hyperparameter tuning"""
    log(f"\nBuilding model for segment: {segment_name} (n={len(segment_df)})")
    
    # Get numeric columns
    numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    # Create a copy of the data with only numeric columns for simplicity
    # In a production environment, you would handle categorical features better
    df_numeric = segment_df[numeric_cols + ['SettlementValue']].copy()
    
    # Get X and y
    X = df_numeric.drop('SettlementValue', axis=1)
    y = df_numeric['SettlementValue']
    
    # Ensure no NaN or infinity values
    valid_indices = np.isfinite(y)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Apply feature selection if requested
    selected_features = None
    feature_selector = None
    
    if select_features and X.shape[1] > 10:  # Only do feature selection if we have enough features
        X_train_selected, selected_features, feature_selector = select_features_for_segment(X_train, y_train, segment_name)
        X_test_selected = feature_selector.transform(X_test)
    else:
        X_train_selected = X_train
        X_test_selected = X_test
        selected_features = X.columns.tolist()
    
    # Apply hyperparameter tuning if requested
    if tune_hyperparams:
        model, best_params = tune_hyperparameters_for_segment(X_train_selected, y_train, segment_name)
    else:
        # Use default XGBoost model
        model = XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train_selected, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    log(f"Segment model performance:")
    log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    
    # Plot feature importance
    if len(selected_features) > 0:
        plt.figure(figsize=(10, 6))
        
        # Get feature importances and match with feature names
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': selected_features[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
            plt.title(f'Feature Importance for {segment_name}')
            plt.tight_layout()
            plt.savefig(f'output/feature_importance_{segment_name.replace(" ", "_").replace("/", "_")}.png')
        
    # Store model metadata
    model_info = {
        'model': model,
        'feature_selector': feature_selector,
        'selected_features': selected_features,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'size': len(segment_df)
    }
    
    return model_info

# Build models for all segments
def build_all_segment_models(segments, tune_hyperparams=True, select_features=True):
    """Build models for all segments"""
    segment_models = {}
    
    for segment_name, segment_df in segments.items():
        model_info = build_model_for_segment(
            segment_df, 
            segment_name,
            tune_hyperparams=tune_hyperparams,
            select_features=select_features
        )
        segment_models[segment_name] = model_info
    
    return segment_models

# Build baseline model
def build_baseline_model(df):
    """Build a baseline model on the entire dataset"""
    log(f"\nBuilding baseline model on entire dataset (n={len(df)})")
    
    # Use the same function but without tuning or feature selection for fair comparison
    return build_model_for_segment(df, "Baseline", tune_hyperparams=False, select_features=False)

# Create meta-features for stacking ensemble
# Here's the fixed create_meta_features function

def create_meta_features(df, segments, segment_models):
    """Create meta-features for stacking ensemble by getting predictions from segment models"""
    log("\nCreating meta-features for stacking ensemble")
    
    # Clone dataframe to avoid modifying original
    df_meta = df.copy()
    
    # Get numeric columns (same as used for segment models)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    # First, identify which column was used for primary segmentation
    # In your case, it should be 'Exceptional_Circumstances'
    primary_segmentation_col = 'Exceptional_Circumstances'  # Default based on your previous results
    
    # Create a column that identifies which segment each row belongs to
    df_meta['segment_id'] = 'Other'  # Default value
    
    # Loop through all segments to populate segment_id
    for segment_name in segment_models.keys():
        if '_' in segment_name:
            # Multi-level segmentation
            parts = segment_name.split('_')
            # Check if the record matches this segment
            mask = (df[primary_segmentation_col] == parts[0])
            for i, part in enumerate(parts[1:], 1):
                # For now, assume 2-level segmentation with 'AccidentType' as second level
                if i == 1 and 'AccidentType' in df.columns:
                    mask = mask & (df['AccidentType'] == part)
            df_meta.loc[mask, 'segment_id'] = segment_name
        else:
            # Single-level segmentation
            mask = df[primary_segmentation_col] == segment_name
            df_meta.loc[mask, 'segment_id'] = segment_name
    
    # Create segment prediction features
    # Initialize all prediction columns with NaN
    for segment_name in segment_models.keys():
        df_meta[f'pred_{segment_name}'] = np.nan
    
    # For each segment, get predictions for records in that segment
    for segment_name, model_info in segment_models.items():
        # Get mask for this segment
        mask = df_meta['segment_id'] == segment_name
        
        # Skip if no matching records
        if not any(mask):
            continue
        
        # Prepare features
        segment_X = df.loc[mask, numeric_cols]
        
        # Apply feature selection if available
        if model_info['feature_selector'] is not None:
            try:
                segment_X = model_info['feature_selector'].transform(segment_X)
            except Exception as e:
                log(f"Error applying feature selection for {segment_name}: {str(e)}")
                # If feature selector fails, use all features
                if model_info['selected_features'] is not None:
                    segment_X = segment_X[model_info['selected_features']]
        elif model_info['selected_features'] is not None:
            # Use selected features
            try:
                segment_X = segment_X[model_info['selected_features']]
            except Exception as e:
                log(f"Error selecting features for {segment_name}: {str(e)}")
                # Continue with all features if selection fails
        
        # Get predictions
        try:
            preds = model_info['model'].predict(segment_X)
            df_meta.loc[mask, f'pred_{segment_name}'] = preds
        except Exception as e:
            log(f"Error getting predictions for segment {segment_name}: {str(e)}")
    
    # Also get predictions from each model for all records
    # This helps the stacked model learn when to trust each segment model
    for segment_name, model_info in segment_models.items():
        try:
            # Prepare features for all records
            all_X = df[numeric_cols]
            
            # Apply feature selection if needed
            if model_info['feature_selector'] is not None:
                try:
                    all_X = model_info['feature_selector'].transform(all_X)
                except:
                    if model_info['selected_features'] is not None:
                        all_X = all_X[model_info['selected_features']]
            elif model_info['selected_features'] is not None:
                try:
                    all_X = all_X[model_info['selected_features']]
                except:
                    pass  # Continue with all features if selection fails
            
            # Get predictions from this model for all records
            all_preds = model_info['model'].predict(all_X)
            df_meta[f'all_pred_{segment_name}'] = all_preds
        except Exception as e:
            log(f"Error getting all predictions from {segment_name} model: {str(e)}")
    
    # Create aggregate prediction features
    pred_cols = [col for col in df_meta.columns if col.startswith('pred_') and not col.startswith('pred_all_')]
    all_pred_cols = [col for col in df_meta.columns if col.startswith('all_pred_')]
    
    # Mean of segment-specific predictions
    if pred_cols:
        df_meta['pred_mean'] = df_meta[pred_cols].mean(axis=1)
    
    # Mean of all model predictions
    if all_pred_cols:
        df_meta['all_pred_mean'] = df_meta[all_pred_cols].mean(axis=1)
    
    # Final feature is the prediction from the record's assigned segment
    df_meta['segment_prediction'] = np.nan
    for segment_name in segment_models.keys():
        mask = df_meta['segment_id'] == segment_name
        if any(mask) and f'pred_{segment_name}' in df_meta.columns:
            df_meta.loc[mask, 'segment_prediction'] = df_meta.loc[mask, f'pred_{segment_name}']
    
    return df_meta

# Build stacked ensemble model
def build_stacked_ensemble(df_meta):
    """Build a stacked ensemble model"""
    log("\nBuilding stacked ensemble model")
    
    # Prepare features for meta-learner
    # Include segment predictions and all numeric features
    meta_cols = [col for col in df_meta.columns if col.startswith('pred_') or 
                 col.startswith('all_pred_') or col == 'segment_prediction']
    
    # Add other important numeric columns
    numeric_cols = df_meta.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue' and 
                    not col.startswith('pred_') and not col.startswith('all_pred_')]
    
    # Make sure we have at least one meta column
    if not meta_cols:
        log("Warning: No prediction columns found in meta-features. Using only numeric features.")
        X_meta = df_meta[numeric_cols].copy()
    else:
        # Filter out columns that are all NaN
        valid_meta_cols = [col for col in meta_cols if not df_meta[col].isna().all()]
        if not valid_meta_cols:
            log("Warning: All prediction columns contain only NaN values. Using only numeric features.")
            X_meta = df_meta[numeric_cols].copy()
        else:
            # Make sure to use .copy() to avoid modifying original data
            X_meta = df_meta[valid_meta_cols + numeric_cols].copy()
    
    # Fill NaN values with 0
    X_meta = X_meta.fillna(0)
    
    # Get target
    y_meta = df_meta['SettlementValue']
    
    # Check for any remaining NaN values
    if X_meta.isna().any().any():
        log("Warning: X_meta still contains NaN values after filling. Filling remaining NaNs with 0.")
        X_meta = X_meta.fillna(0)
    
    if y_meta.isna().any():
        log("Warning: y_meta contains NaN values. Dropping corresponding rows.")
        valid_indices = ~y_meta.isna()
        X_meta = X_meta.loc[valid_indices]
        y_meta = y_meta.loc[valid_indices]
    
    # Convert to numpy arrays to avoid any dtype issues with XGBoost
    X_meta_np = X_meta.to_numpy()
    y_meta_np = y_meta.to_numpy()
    
    log(f"X_meta shape: {X_meta.shape}, y_meta shape: {y_meta.shape}")
    
    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta_np, y_meta_np, test_size=0.2, random_state=42
    )
    
    # Create meta-learner (using XGBoost with light hyperparameters)
    meta_model = XGBRegressor(
        learning_rate=0.05,
        max_depth=4,
        n_estimators=150,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    
    try:
        # Train meta-model
        meta_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = meta_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        log(f"Stacked ensemble performance:")
        log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        # Store the feature names for plotting
        feature_names = X_meta.columns.tolist()
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': meta_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot top 15 features or all if less than 15
        num_features = min(15, len(feature_importance))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(num_features))
        plt.title('Feature Importance for Stacked Ensemble')
        plt.tight_layout()
        plt.savefig('output/feature_importance_stacked_ensemble.png')
        
        return meta_model, {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    except Exception as e:
        log(f"Error in stacked ensemble building: {str(e)}")
        import traceback
        log(traceback.format_exc())
        
        # Return a simple fallback model
        log("Creating a simple fallback model instead")
        
        # Use only key numeric features
        fallback_X = df_meta.select_dtypes(include=['float64', 'int64']).drop('SettlementValue', axis=1)
        fallback_y = df_meta['SettlementValue']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            fallback_X, fallback_y, test_size=0.2, random_state=42
        )
        
        # Simple XGBoost model
        fallback_model = XGBRegressor(random_state=42)
        
        # Fit
        fallback_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = fallback_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        log(f"Fallback model performance:")
        log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return fallback_model, {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

# Compare models
def compare_and_visualize(baseline_metrics, segment_models, stacked_metrics):
    """Compare model performances and create visualizations"""
    # Collect metrics
    segment_metrics = {name: info['metrics'] for name, info in segment_models.items()}
    segment_sizes = {name: info['size'] for name, info in segment_models.items()}
    
    # Calculate weighted average for segmented models
    total_size = sum(segment_sizes.values())
    weighted_rmse = sum(segment_metrics[seg]['rmse'] * segment_sizes[seg] / total_size for seg in segment_metrics)
    weighted_mae = sum(segment_metrics[seg]['mae'] * segment_sizes[seg] / total_size for seg in segment_metrics)
    weighted_r2 = sum(segment_metrics[seg]['r2'] * segment_sizes[seg] / total_size for seg in segment_metrics)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Baseline': [
            baseline_metrics['rmse'],
            baseline_metrics['mae'],
            baseline_metrics['r2']
        ],
        'Segmented (Weighted)': [
            weighted_rmse,
            weighted_mae,
            weighted_r2
        ],
        'Stacked Ensemble': [
            stacked_metrics['rmse'],
            stacked_metrics['mae'],
            stacked_metrics['r2']
        ]
    }, index=['RMSE', 'MAE', 'R2'])
    
    log("\nModel Comparison:")
    log(comparison.to_string())
    
    # Calculate improvements over baseline
    baseline_r2 = baseline_metrics['r2']
    improvements = pd.DataFrame({
        'Model': ['Segmented (Weighted)', 'Stacked Ensemble'],
        'R2 Score': [weighted_r2, stacked_metrics['r2']],
        'Improvement (%)': [
            (weighted_r2 - baseline_r2) / baseline_r2 * 100,
            (stacked_metrics['r2'] - baseline_r2) / baseline_r2 * 100
        ]
    })
    
    log("\nImprovements over Baseline:")
    log(improvements.to_string())
    
    # Create visualizations
    
    # R2 comparison
    plt.figure(figsize=(10, 6))
    comparison.loc['R2'].plot(kind='bar')
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.ylim(0.8, 1.0)  # Adjusted for better visualization
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/enhanced_r2_comparison.png')
    
    # RMSE comparison
    plt.figure(figsize=(10, 6))
    comparison.loc['RMSE'].plot(kind='bar')
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/enhanced_rmse_comparison.png')
    
    # Individual segment performance
    segment_performance = pd.DataFrame({
        'Segment': list(segment_metrics.keys()),
        'R2': [segment_metrics[seg]['r2'] for seg in segment_metrics],
        'RMSE': [segment_metrics[seg]['rmse'] for seg in segment_metrics],
        'Size': [segment_sizes[seg] for seg in segment_metrics]
    }).sort_values('R2', ascending=False)
    
    log("\nPerformance by Segment:")
    log(segment_performance.to_string())
    
    # Plot segment R2 scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='R2', y='Segment', data=segment_performance)
    plt.title('R² Score by Segment')
    plt.xlabel('R² Score')
    plt.xlim(0.7, 1.0)  # Adjusted for better visualization
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/enhanced_segment_r2.png')
    
    # Scatter plot of segment size vs R2
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Size', y='R2', data=segment_performance, s=100)
    
    # Add labels for each point
    for i, row in segment_performance.iterrows():
        plt.text(row['Size'], row['R2'], row['Segment'], fontsize=8)
    
    plt.title('Segment Size vs R² Score')
    plt.xlabel('Segment Size (Number of Records)')
    plt.ylabel('R² Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/enhanced_segment_size_vs_r2.png')
    
    return comparison, improvements, segment_performance

# Save models and results
def save_results(baseline_model, segment_models, stacked_model, comparison, improvements, segment_performance):
    """Save models and results"""
    # Create results directory
    os.makedirs('output/models', exist_ok=True)
    
    # Save the baseline model
    joblib.dump(baseline_model['model'], 'output/models/baseline_model.pkl')
    
    # Save segment models
    for segment_name, model_info in segment_models.items():
        safe_name = segment_name.replace(" ", "_").replace("/", "_")
        joblib.dump(model_info['model'], f'output/models/{safe_name}_model.pkl')
    
    # Save stacked model
    joblib.dump(stacked_model, 'output/models/stacked_model.pkl')
    
    # Save results to CSV
    comparison.to_csv('output/enhanced_model_comparison.csv')
    improvements.to_csv('output/enhanced_model_improvements.csv')
    segment_performance.to_csv('output/enhanced_segment_performance.csv')
    
    log("\nAll models and results saved to output directory")

# Main function
def main():
    start_time = time.time()
    
    try:
        log("======= STARTING ENHANCED MODEL BUILDING PROCESS =======")
        log(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        df = load_data('Synthetic_Data_For_Students.csv')
        
        # Preprocess data
        df = preprocess_data(df)
        
        # Engineer features
        df = engineer_features(df)
        
        # Build baseline model
        baseline_model = build_baseline_model(df)
        
        # Create multi-level segmentation
        segments = create_multi_segments(df, 'Exceptional_Circumstances', 'AccidentType')
        
        # Build models for all segments (tune = True, feature selection = True)
        segment_models = build_all_segment_models(segments, tune_hyperparams=True, select_features=True)
        
        # Create meta-features for stacking
        df_meta = create_meta_features(df, segments, segment_models)
        
        # Build stacked ensemble model
        stacked_model, stacked_metrics = build_stacked_ensemble(df_meta)
        
        # Compare and visualize results
        comparison, improvements, segment_performance = compare_and_visualize(
            baseline_model['metrics'], 
            segment_models, 
            stacked_metrics
        )
        
        # Save all models and results
        save_results(
            baseline_model,
            segment_models, 
            stacked_model,
            comparison, 
            improvements, 
            segment_performance
        )
        
        # Print conclusion
        log("\n" + "="*50)
        log("MODEL BUILDING COMPLETE")
        log("="*50)
        
        # Find best model
        if stacked_metrics['r2'] > baseline_model['metrics']['r2']:
            improvement = (stacked_metrics['r2'] - baseline_model['metrics']['r2']) / baseline_model['metrics']['r2'] * 100
            log(f"\nBest model: Stacked Ensemble")
            log(f"R² Score: {stacked_metrics['r2']:.4f}")
            log(f"Improvement over baseline: {improvement:.2f}%")
        else:
            log(f"\nStacked model did not improve over baseline")
        
        # Print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        log(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        
    except Exception as e:
        log(f"\nERROR in main execution: {str(e)}")
        import traceback
        log(traceback.format_exc())
    finally:
        # Close the log file
        log("\n======= MODEL BUILDING PROCESS FINISHED =======")
        log_file.close()

if __name__ == "__main__":
    main()