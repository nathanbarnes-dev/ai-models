import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, LassoCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('output', exist_ok=True)

# Create a log file
log_filename = f'output/unified_segmented_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
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

# Preprocess data with advanced cleaning and normalization
def preprocess_data(df):
    """Preprocess the data for modeling with advanced cleaning"""
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
    
    # Handle outliers in numeric predictors using IQR
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # Fill missing values in numeric columns with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing values in categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Log column types for reference
    log(f"Numeric columns: {len(numeric_cols)}")
    log(f"Categorical columns: {len(categorical_cols)}")
    
    return df

# Enhanced feature engineering with domain-specific features
def engineer_features(df):
    """Add engineered features with domain knowledge to improve model performance"""
    log("\n--- Adding Enhanced Engineered Features ---")
    
    # Create a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Count of engineered features added
    feature_count = 0
    
    # Time between accident and claim filing
    if 'Accident Date' in df.columns and 'Claim Date' in df.columns:
        try:
            df_new['Days_To_Claim'] = (pd.to_datetime(df_new['Claim Date']) - 
                                      pd.to_datetime(df_new['Accident Date'])).dt.days
            # Create bucketed version
            df_new['Days_To_Claim_Bucket'] = pd.cut(df_new['Days_To_Claim'], 
                                                   bins=[-1, 7, 30, 90, 365, float('inf')],
                                                   labels=['1_Week', '1_Month', '3_Months', '1_Year', 'Over_1_Year'])
            feature_count += 2
        except:
            log("Could not calculate days between dates")
    
    # Total special damages (sum of all Special* columns)
    special_columns = [col for col in df.columns if col.startswith('Special')]
    if len(special_columns) > 0:
        df_new['Total_Special_Damages'] = df_new[special_columns].sum(axis=1)
        # Log transform for skewed monetary values
        df_new['Log_Special_Damages'] = np.log1p(df_new['Total_Special_Damages'])
        feature_count += 2
    
    # Total general damages
    general_columns = [col for col in df.columns if col.startswith('General')]
    if len(general_columns) > 0:
        df_new['Total_General_Damages'] = df_new[general_columns].sum(axis=1)
        # Log transform
        df_new['Log_General_Damages'] = np.log1p(df_new['Total_General_Damages'])
        feature_count += 2
    
    # Total damages
    if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
        df_new['Total_Damages'] = df_new['Total_Special_Damages'] + df_new['Total_General_Damages']
        df_new['Log_Total_Damages'] = np.log1p(df_new['Total_Damages'])
        feature_count += 2
    
    # Ratio features
    if 'Total_Special_Damages' in df_new.columns and 'Total_General_Damages' in df_new.columns:
        # Avoid division by zero
        df_new['Special_General_Ratio'] = df_new['Total_Special_Damages'] / df_new['Total_General_Damages'].replace(0, 1)
        # Create buckets of the ratio for non-linear relationships
        df_new['Special_General_Bucket'] = pd.cut(df_new['Special_General_Ratio'], 
                                                 bins=[-1, 0.5, 1.0, 2.0, 5.0, float('inf')],
                                                 labels=['Very_Low', 'Low', 'Even', 'High', 'Very_High'])
        feature_count += 2
    
    # Age-related features
    if 'Driver Age' in df_new.columns:
        # Age buckets for non-linear age effects
        df_new['Age_Bucket'] = pd.cut(df_new['Driver Age'], 
                                     bins=[0, 25, 35, 50, 65, 100],
                                     labels=['Young', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior'])
        
        # Young or old driver flag (higher risk groups)
        df_new['Age_Risk_Group'] = ((df_new['Driver Age'] < 25) | (df_new['Driver Age'] > 65)).astype(int)
        feature_count += 2
        
        # Age-damage interaction
        if 'Total_Damages' in df_new.columns:
            df_new['Age_Damage_Interaction'] = df_new['Driver Age'] * df_new['Total_Damages'] / 1000
            feature_count += 1
    
    # Create categorical flags and enhanced indicators
    
    # Injury severity indicators
    injury_flag_created = False
    for col in df_new.select_dtypes(include=['object']).columns:
        if 'Injury' in col or 'injury' in col:
            # Check multiple severity levels
            if not injury_flag_created:
                df_new['Severe_Injury_Flag'] = df_new[col].str.lower().str.contains('severe|major|critical', na=False).astype(int)
                df_new['Moderate_Injury_Flag'] = df_new[col].str.lower().str.contains('moderate|significant', na=False).astype(int)
                df_new['Minor_Injury_Flag'] = df_new[col].str.lower().str.contains('minor|slight|small', na=False).astype(int)
                injury_flag_created = True
                feature_count += 3
    
    # Whiplash specific flag (common in car accidents)
    if 'Whiplash' in df_new.columns:
        df_new['Whiplash_Numeric'] = (df_new['Whiplash'] == 'Yes').astype(int)
        feature_count += 1
    
    # Multi-injury indicator (multiple types of injuries)
    injury_columns = [col for col in df_new.columns if 'injury' in col.lower() or 'Injury' in col]
    if len(injury_columns) >= 2:
        try:
            df_new['Multi_Injury_Count'] = df_new[injury_columns].apply(
                lambda row: sum(1 for val in row if isinstance(val, str) and val.lower() not in ['no', 'none', 'unknown', '']), 
                axis=1
            )
            df_new['Has_Multiple_Injuries'] = (df_new['Multi_Injury_Count'] > 1).astype(int)
            feature_count += 2
        except:
            log("Could not create multi-injury indicators")
    
    # Create complex interaction features for severe cases
    if 'Severe_Injury_Flag' in df_new.columns:
        # Severe injury interactions with damages
        if 'Total_Special_Damages' in df_new.columns:
            df_new['Severe_Injury_Special_Damages'] = df_new['Severe_Injury_Flag'] * df_new['Total_Special_Damages']
            feature_count += 1
        
        if 'Total_General_Damages' in df_new.columns:
            df_new['Severe_Injury_General_Damages'] = df_new['Severe_Injury_Flag'] * df_new['Total_General_Damages']
            feature_count += 1
    
    # Psychological injury impacts
    if 'Minor_Psychological_Injury' in df_new.columns:
        df_new['Psych_Injury_Numeric'] = (df_new['Minor_Psychological_Injury'] == 'Yes').astype(int)
        
        # Interaction with damages
        if 'Total_Damages' in df_new.columns:
            df_new['Psych_Damage_Interaction'] = df_new['Psych_Injury_Numeric'] * df_new['Total_Damages']
            feature_count += 2
    
    # Accident type specific features
    if 'AccidentType' in df_new.columns:
        # Rear-end collision flag (common in insurance)
        df_new['Is_Rear_End'] = df_new['AccidentType'].str.contains('Rear end', na=False).astype(int)
        
        # Multi-vehicle collision
        df_new['Is_Multi_Vehicle'] = df_new['AccidentType'].str.contains('3 car|multiple', na=False).astype(int)
        feature_count += 2
    
    # Polynomial features for key numeric columns
    for col in ['Total_Special_Damages', 'Total_General_Damages', 'Total_Damages']:
        if col in df_new.columns:
            df_new[f'{col}_Squared'] = df_new[col] ** 2
            feature_count += 1
    
    # Create combined categorical features
    if 'AccidentType' in df_new.columns and 'Exceptional_Circumstances' in df_new.columns:
        # Combined feature that captures both accident type and circumstances
        df_new['Accident_Circumstances'] = df_new['AccidentType'] + '_' + df_new['Exceptional_Circumstances']
        feature_count += 1
    
    log(f"Added {feature_count} engineered features")
    return df_new

# Create optimized multi-level segmentation 
def create_optimized_segments(df, segmentation_cols=['Exceptional_Circumstances', 'AccidentType'], min_segment_size=50):
    """Create optimized segments based on the most predictive categorical variables"""
    log(f"\n--- Creating Optimized Multi-level Segmentation ---")
    
    # Convert segmentation_cols to list if it's a string
    if isinstance(segmentation_cols, str):
        segmentation_cols = [segmentation_cols]
    
    segments = {}
    
    # Get unique combinations of segment values
    if len(segmentation_cols) == 1:
        # Single column segmentation
        primary_col = segmentation_cols[0]
        segment_groups = df.groupby(primary_col)
        
        for segment_value, segment_df in segment_groups:
            if len(segment_df) >= min_segment_size:
                segments[segment_value] = segment_df
                log(f"Created segment '{segment_value}' with {len(segment_df)} rows")
    
    else:
        # Multi-column segmentation
        # First, create a combined column for segmentation
        df = df.copy()
        df['_segment_id'] = ''
        
        for col in segmentation_cols:
            df['_segment_id'] += df[col].astype(str) + '_'
        
        # Group by the combined segment ID
        segment_groups = df.groupby('_segment_id')
        
        for segment_value, segment_df in segment_groups:
            if len(segment_df) >= min_segment_size:
                segments[segment_value] = segment_df.drop('_segment_id', axis=1)
                log(f"Created segment '{segment_value}' with {len(segment_df)} rows")
    
    # Identify segments too small for individual modeling
    small_segments = []
    for segment_value, segment_df in list(segments.items()):
        if len(segment_df) < min_segment_size:
            small_segments.append(segment_df)
            del segments[segment_value]
            log(f"Removed segment '{segment_value}' with only {len(segment_df)} rows")
    
    # Combine small segments into an "Other" category if needed
    if small_segments:
        other_df = pd.concat(small_segments)
        
        if len(other_df) >= min_segment_size:
            segments['Other'] = other_df
            log(f"Created 'Other' segment with {len(other_df)} rows from {len(small_segments)} small segments")
        else:
            # If "Other" is still too small, find closest large segment to merge with
            log(f"'Other' segment with {len(other_df)} rows is too small, will merge with appropriate segment")
            
            # Find largest segment to merge with
            largest_segment = max(segments.items(), key=lambda x: len(x[1]))
            segments[largest_segment[0]] = pd.concat([largest_segment[1], other_df])
            log(f"Merged small segments with '{largest_segment[0]}', now has {len(segments[largest_segment[0]])} rows")
    
    # Create second-level segments for large segments
    large_segments = {k: v for k, v in segments.items() if len(v) >= 3 * min_segment_size}
    
    if large_segments and len(segmentation_cols) == 1 and len(df.columns) > 10:
        # Try to find a secondary segmentation column through correlation analysis
        log("Analyzing large segments for possible sub-segmentation")
        
        # Get all categorical columns except the primary segmentation column
        categorical_cols = [col for col in df.select_dtypes(include=['object']).columns 
                           if col != segmentation_cols[0]]
        
        # For each large segment, try to find a good secondary segmentation column
        for segment_name, segment_df in large_segments.items():
            best_subsegments = None
            best_subsegment_count = 0
            best_col = None
            
            for col in categorical_cols:
                # Skip columns with too many unique values
                if segment_df[col].nunique() > 10:
                    continue
                
                # Try segmenting by this column
                test_subsegments = {}
                for val, sub_df in segment_df.groupby(col):
                    if len(sub_df) >= min_segment_size:
                        test_subsegments[f"{segment_name}_{val}"] = sub_df
                
                # If this creates more usable subsegments, keep it
                if len(test_subsegments) > best_subsegment_count:
                    best_subsegments = test_subsegments
                    best_subsegment_count = len(test_subsegments)
                    best_col = col
            
            # If we found good subsegments, use them
            if best_subsegments and best_subsegment_count >= 2:
                log(f"Sub-segmenting '{segment_name}' by '{best_col}' into {best_subsegment_count} segments")
                
                # Remove the original large segment
                del segments[segment_name]
                
                # Add the subsegments
                for subseg_name, subseg_df in best_subsegments.items():
                    segments[subseg_name] = subseg_df
                    log(f"Created sub-segment '{subseg_name}' with {len(subseg_df)} rows")
    
    log(f"Final segmentation created {len(segments)} segments")
    
    # Create a segment mapping dictionary to use later for prediction
    segment_map = {}
    for segment_name, segment_df in segments.items():
        for idx in segment_df.index:
            segment_map[idx] = segment_name
    
    return segments, segment_map

# Enhanced feature selection using multiple methods
def advanced_feature_selection(X, y, segment_name, method='combined'):
    """Select important features using multiple methods for robustness"""
    log(f"Advanced feature selection for segment: {segment_name} using {method} method")
    
    if X.shape[1] <= 5:
        log(f"Only {X.shape[1]} features available, skipping feature selection")
        return X, X.columns.tolist(), None
    
    if method == 'xgboost':
        # Use XGBoost feature importance
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
        
    elif method == 'lasso':
        # Use Lasso for feature selection
        feature_selector = LassoCV(
            cv=5, 
            random_state=42,
            max_iter=1000
        )
        
        # Fit the model
        feature_selector.fit(X, y)
        
        # Create selector based on coefficients
        selection_model = SelectFromModel(
            feature_selector, 
            threshold='mean',
            prefit=True
        )
        
    elif method == 'rfe':
        # Use Recursive Feature Elimination
        estimator = RandomForestRegressor(
            n_estimators=100, 
            random_state=42
        )
        
        # Keep at least half the features or 10, whichever is smaller
        n_features_to_select = max(min(X.shape[1] // 2, 10), 3)
        
        feature_selector = RFE(
            estimator=estimator, 
            n_features_to_select=n_features_to_select,
            step=1
        )
        
        # Fit the model
        feature_selector.fit(X, y)
        
        # RFE itself is the selection model
        selection_model = feature_selector
        
    elif method == 'combined':
        # Use multiple methods and combine their results
        # First, get feature importance from XGBoost
        xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X, y)
        xgb_importances = xgb_model.feature_importances_
        
        # Second, get coefficients from Lasso
        lasso_model = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso_model.fit(X, y)
        lasso_importances = np.abs(lasso_model.coef_)
        
        # Normalize both to 0-1 range
        xgb_importances = xgb_importances / np.max(xgb_importances)
        if np.max(lasso_importances) > 0:
            lasso_importances = lasso_importances / np.max(lasso_importances)
        
        # Combine importances (average)
        combined_importances = (xgb_importances + lasso_importances) / 2
        
        # Create a threshold at mean importance
        threshold = np.mean(combined_importances)
        
        # Select features above threshold
        selected_indices = combined_importances > threshold
        
        # Ensure we keep at least 3 features
        if sum(selected_indices) < 3:
            # Keep top 3 features
            top_indices = np.argsort(combined_importances)[-3:]
            selected_indices = np.zeros_like(selected_indices, dtype=bool)
            selected_indices[top_indices] = True
        
        # Get selected feature names
        selected_features = X.columns[selected_indices].tolist()
        
        log(f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
        
        # Return directly since we're not using SelectFromModel
        return X[selected_features], selected_features, selected_indices
    
    else:
        # Default to XGBoost if method not recognized
        log(f"Method '{method}' not recognized, defaulting to XGBoost")
        return advanced_feature_selection(X, y, segment_name, method='xgboost')
    
    # Get selected features
    if hasattr(selection_model, 'get_support'):
        selected_indices = selection_model.get_support()
    else:
        selected_indices = selection_model.support_
    
    selected_features = X.columns[selected_indices].tolist()
    
    log(f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
    
    # Transform X to select features
    X_selected = X[selected_features]
    
    return X_selected, selected_features, selection_model

# Tune hyperparameters with more focused grid
def advanced_hyperparameter_tuning(X, y, segment_name):
    """Tune XGBoost hyperparameters with focused grid and early stopping"""
    log(f"Advanced hyperparameter tuning for segment: {segment_name}")
    
    # Define parameter grid - focused on most important parameters
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1]
    }
    
    # For very small segments, use a simplified grid
    if X.shape[0] < 200:
        log(f"Small segment detected ({X.shape[0]} samples), using simplified parameter grid")
        param_grid = {
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [100],
            'subsample': [0.8]
        }
    
    # Create evaluation set for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    eval_set = [(X_val, y_val)]
    
    # Create XGBoost model with early stopping
    xgb_model = XGBRegressor(
        random_state=42,
        early_stopping_rounds=10
    )
    
    # Create cross-validation strategy
    cv = KFold(n_splits=min(5, X.shape[0] // 10), shuffle=True, random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit GridSearchCV with early stopping
    grid_search.fit(
        X_train, 
        y_train, 
        eval_set=eval_set, 
        verbose=False
    )
    
    # Get best parameters
    best_params = grid_search.best_params_
    log(f"Best hyperparameters: {best_params}")
    
    # Create model with best parameters
    best_model = XGBRegressor(
        random_state=42,
        **best_params
    )
    
    # Train on full dataset
    best_model.fit(X, y)
    
    return best_model, best_params

# Build optimized model for a segment with robust evaluation
def build_optimized_model(segment_df, segment_name, tune_hyperparams=True, select_features=True):
    """Build an optimized model for a single segment with robust evaluation"""
    log(f"\nBuilding optimized model for segment: {segment_name} (n={len(segment_df)})")
    
    # Get numeric columns
    numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    # Create a copy of the data with only numeric columns
    # In a production version, we would properly encode categorical variables
    df_numeric = segment_df[numeric_cols + ['SettlementValue']].copy()
    
    # Log potential issues
    if df_numeric.isnull().any().any():
        log(f"Warning: Numeric data contains {df_numeric.isnull().sum().sum()} missing values, filling with medians")
        df_numeric = df_numeric.fillna(df_numeric.median())
    
    # Get X and y
    X = df_numeric.drop('SettlementValue', axis=1)
    y = df_numeric['SettlementValue']
    
    # Ensure no NaN or infinity values
    valid_indices = np.isfinite(y)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Apply feature selection if requested
    selected_features = None
    feature_selector = None
    
    if select_features and X.shape[1] > 5:  # Only do feature selection if we have enough features
        X_selected, selected_features, feature_selector = advanced_feature_selection(X, y, segment_name, method='combined')
    else:
        X_selected = X
        selected_features = X.columns.tolist()
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Apply hyperparameter tuning if requested
    if tune_hyperparams:
        model, best_params = advanced_hyperparameter_tuning(X_train, y_train, segment_name)
    else:
        # Use default XGBoost model
        model = XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except:
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    log(f"Segment model performance:")
    log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
    
    # Plot feature importance
    if hasattr(model, 'feature_importances_') and len(selected_features) > 0:
        plt.figure(figsize=(10, 6))
        
        # Get feature importances
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': selected_features[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title(f'Feature Importance for {segment_name}')
        plt.tight_layout()
        plt.savefig(f'output/optimized_importance_{segment_name.replace(" ", "_").replace("/", "_")[:30]}.png')
    
    # Store model metadata
    model_info = {
        'model': model,
        'feature_selector': feature_selector,
        'selected_features': selected_features,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        },
        'size': len(segment_df)
    }
    
    return model_info

# Build models for all segments
def build_all_segment_models(segments, tune_hyperparams=True, select_features=True):
   """Build models for all segments"""
   segment_models = {}
   
   for segment_name, segment_df in segments.items():
       model_info = build_optimized_model(
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
   return build_optimized_model(df, "Baseline", tune_hyperparams=False, select_features=False)

# Create enhanced meta-features for advanced stacking ensemble
def create_enhanced_meta_features(df, segment_map, segment_models):
   """Create enhanced meta-features for stacking ensemble with additional information"""
   log("\nCreating enhanced meta-features for stacking ensemble")
   
   # Create a copy of the dataframe to avoid modifying the original
   df_meta = df.copy()
   
   # Get numeric columns used in segment models
   numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
   numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
   
   # Create a column that indicates which segment each row belongs to
   segment_ids = []
   for idx in df_meta.index:
       if idx in segment_map:
           segment_ids.append(segment_map[idx])
       else:
           segment_ids.append('Unknown')
   
   df_meta['segment_id'] = segment_ids
   
   # Create segment prediction features (each segment model predicts for its rows)
   for segment_name, model_info in segment_models.items():
       # Initialize prediction column
       col_name = f'pred_{segment_name}'
       df_meta[col_name] = np.nan
       
       # Get rows for this segment
       segment_mask = df_meta['segment_id'] == segment_name
       segment_rows = df_meta.loc[segment_mask]
       
       if len(segment_rows) > 0:
           # Prepare features 
           segment_X = segment_rows[numeric_cols].copy()
           
           # Apply feature selection if available
           if model_info['feature_selector'] is not None:
               try:
                   # If SelectFromModel was used
                   if hasattr(model_info['feature_selector'], 'transform'):
                       segment_X_selected = model_info['feature_selector'].transform(segment_X)
                       # We need to use the model directly since we're passing numpy array
                       segment_preds = model_info['model'].predict(segment_X_selected)
                   else:
                       # If we just have column names
                       selected_features = model_info['selected_features']
                       segment_X_selected = segment_X[selected_features]
                       segment_preds = model_info['model'].predict(segment_X_selected)
               except Exception as e:
                   log(f"Error applying feature selection for {segment_name}: {str(e)}")
                   # Fallback to using all features
                   segment_preds = model_info['model'].predict(segment_X)
           else:
               # Just use the model directly if no feature selection
               segment_preds = model_info['model'].predict(segment_X)
           
           # Store predictions
           df_meta.loc[segment_mask, col_name] = segment_preds
   
   # Create cross-segment predictions (each model predicts for all rows)
   # This helps the meta-model learn when to trust each segment model
   for segment_name, model_info in segment_models.items():
       try:
           # Get all rows
           all_X = df_meta[numeric_cols].copy()
           
           # Apply feature selection
           if model_info['feature_selector'] is not None:
               try:
                   if hasattr(model_info['feature_selector'], 'transform'):
                       all_X_selected = model_info['feature_selector'].transform(all_X)
                       all_preds = model_info['model'].predict(all_X_selected)
                   else:
                       selected_features = model_info['selected_features']
                       all_X_selected = all_X[selected_features]
                       all_preds = model_info['model'].predict(all_X_selected)
               except:
                   # Fallback
                   all_preds = model_info['model'].predict(all_X)
           else:
               # No feature selection
               all_preds = model_info['model'].predict(all_X)
           
           # Store cross-segment predictions
           df_meta[f'cross_pred_{segment_name}'] = all_preds
           
       except Exception as e:
           log(f"Error creating cross-predictions for {segment_name}: {str(e)}")
   
   # Create advanced meta-features
   
   # 1. Primary segment prediction (from the model assigned to that segment)
   df_meta['primary_segment_pred'] = np.nan
   for segment_name in segment_models.keys():
       mask = df_meta['segment_id'] == segment_name
       if any(mask) and f'pred_{segment_name}' in df_meta.columns:
           df_meta.loc[mask, 'primary_segment_pred'] = df_meta.loc[mask, f'pred_{segment_name}']
   
   # 2. Average of all segment predictions
   pred_cols = [col for col in df_meta.columns if col.startswith('pred_')]
   if pred_cols:
       df_meta['avg_segment_pred'] = df_meta[pred_cols].mean(axis=1)
   
   # 3. Average of cross-segment predictions
   cross_pred_cols = [col for col in df_meta.columns if col.startswith('cross_pred_')]
   if cross_pred_cols:
       df_meta['avg_cross_pred'] = df_meta[cross_pred_cols].mean(axis=1)
   
   # 4. Weighted average based on segment size
   if cross_pred_cols:
       segment_sizes = {name: info['size'] for name, info in segment_models.items()}
       total_size = sum(segment_sizes.values())
       segment_weights = {name: size / total_size for name, size in segment_sizes.items()}
       
       # Apply weights to predictions
       df_meta['weighted_pred'] = 0
       for segment_name, weight in segment_weights.items():
           col_name = f'cross_pred_{segment_name}'
           if col_name in df_meta.columns:
               df_meta['weighted_pred'] += df_meta[col_name] * weight
   
   # 5. Variance of predictions (uncertainty measure)
   if cross_pred_cols:
       df_meta['pred_variance'] = df_meta[cross_pred_cols].var(axis=1)
   
   # 6. Min and max predictions (bounds)
   if cross_pred_cols:
       df_meta['min_pred'] = df_meta[cross_pred_cols].min(axis=1)
       df_meta['max_pred'] = df_meta[cross_pred_cols].max(axis=1)
   
   # 7. Segment-based features
   df_meta['segment_size'] = df_meta['segment_id'].map(
       {name: info['size'] for name, info in segment_models.items()}
   ).fillna(0)
   
   df_meta['segment_r2'] = df_meta['segment_id'].map(
       {name: info['metrics']['r2'] for name, info in segment_models.items()}
   ).fillna(0)
   
   # Fill any remaining NaN values with appropriate defaults
   pred_cols = [col for col in df_meta.columns if 'pred' in col]
   for col in pred_cols:
       if df_meta[col].isna().any():
           if 'avg' in col or 'weighted' in col:
               # For average columns, fill with primary segment prediction
               df_meta[col] = df_meta[col].fillna(df_meta['primary_segment_pred'])
           else:
               # For other prediction columns, fill with the dataset average
               df_meta[col] = df_meta[col].fillna(df['SettlementValue'].mean())
   
   # Remove the segment_id column as it's categorical and will be processed later
   meta_features = df_meta.drop('segment_id', axis=1)
   
   log(f"Created {len(meta_features.columns) - len(numeric_cols) - 1} meta-features")
   
   return meta_features, df_meta['segment_id']

# Build advanced stacked ensemble
def build_advanced_stacked_ensemble(meta_features, target, segment_ids):
   """Build an advanced stacked ensemble with optimized meta-learner"""
   log("\nBuilding advanced stacked ensemble")
   
   # Prepare meta-features
   # Include all predictions and created meta-features
   meta_cols = [col for col in meta_features.columns if 'pred' in col or 
                col in ['segment_size', 'segment_r2', 'pred_variance', 'min_pred', 'max_pred']]
   
   # Add a few important original features
   important_features = [col for col in meta_features.columns if 
                        col in ['Total_Special_Damages', 'Total_General_Damages', 'Total_Damages',
                              'Log_Total_Damages', 'Special_General_Ratio'] and
                        col not in meta_cols]
   
   # Combine meta-features and important original features
   X_meta = meta_features[meta_cols + important_features].copy()
   
   # Remove highly correlated features to reduce multicollinearity
   # Calculate correlation matrix
   corr_matrix = X_meta.corr().abs()
   
   # Get upper triangle of correlation matrix
   upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
   
   # Find features with correlation greater than 0.95
   to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
   
   if to_drop:
       log(f"Dropping {len(to_drop)} highly correlated features: {to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")
       X_meta = X_meta.drop(to_drop, axis=1)
   
   # Fill any remaining NaN values
   X_meta = X_meta.fillna(0)
   
   # Get target variable
   y_meta = target
   
   # Split for evaluation
   X_train, X_test, y_train, y_test = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)
   
   # Build a meta-learner using a stacking approach
   # First, create base models for stacking
   base_models = [
       ('xgb', XGBRegressor(
           learning_rate=0.05,
           max_depth=4,
           n_estimators=150, 
           random_state=42
       )),
       ('gbm', GradientBoostingRegressor(
           learning_rate=0.05,
           max_depth=4,
           n_estimators=150,
           random_state=42
       )),
       ('rf', RandomForestRegressor(
           n_estimators=150,
           max_depth=8,
           random_state=42
       ))
   ]
   
   # Create final estimator (meta-model)
   final_estimator = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)
   
   # Create stacking regressor
   stacking_regressor = StackingRegressor(
       estimators=base_models,
       final_estimator=final_estimator,
       cv=5,
       n_jobs=-1
   )
   
   # Train the stacked model
   stacking_regressor.fit(X_train, y_train)
   
   # Evaluate
   y_pred = stacking_regressor.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   rmse = np.sqrt(mse)
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   
   try:
       mape = mean_absolute_percentage_error(y_test, y_pred)
   except:
       mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
   
   log(f"Advanced stacked ensemble performance:")
   log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
   
   # Evaluate feature importance via permutation importance
   try:
       from sklearn.inspection import permutation_importance
       
       # Calculate permutation importance
       perm_importance = permutation_importance(
           stacking_regressor, X_test, y_test, n_repeats=10, random_state=42
       )
       
       # Create DataFrame for visualization
       importance_df = pd.DataFrame({
           'Feature': X_meta.columns,
           'Importance': perm_importance.importances_mean
       }).sort_values('Importance', ascending=False)
       
       # Plot importance
       plt.figure(figsize=(12, 8))
       sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
       plt.title('Feature Importance for Advanced Stacked Ensemble')
       plt.tight_layout()
       plt.savefig('output/advanced_stacked_ensemble_importance.png')
       
       # Log top features
       log("\nTop features in stacked ensemble:")
       for idx, row in importance_df.head(10).iterrows():
           log(f"  {row['Feature']}: {row['Importance']:.6f}")
           
   except Exception as e:
       log(f"Could not calculate permutation importance: {str(e)}")
   
   return stacking_regressor, {
       'mse': mse,
       'rmse': rmse,
       'mae': mae,
       'r2': r2,
       'mape': mape
   }

# Comprehensive model comparison
def compare_models(baseline_metrics, segment_models, stacked_metrics):
   """Comprehensive comparison of model performance with visualizations"""
   # Collect metrics
   segment_metrics = {name: info['metrics'] for name, info in segment_models.items()}
   segment_sizes = {name: info['size'] for name, info in segment_models.items()}
   
   # Calculate weighted average for segmented models
   total_size = sum(segment_sizes.values())
   weighted_rmse = sum(segment_metrics[seg]['rmse'] * segment_sizes[seg] / total_size for seg in segment_metrics)
   weighted_mae = sum(segment_metrics[seg]['mae'] * segment_sizes[seg] / total_size for seg in segment_metrics)
   weighted_r2 = sum(segment_metrics[seg]['r2'] * segment_sizes[seg] / total_size for seg in segment_metrics)
   
   try:
       weighted_mape = sum(segment_metrics[seg]['mape'] * segment_sizes[seg] / total_size for seg in segment_metrics)
   except:
       weighted_mape = 0
   
   # Create comparison dataframe
   comparison = pd.DataFrame({
       'Baseline': [
           baseline_metrics['rmse'],
           baseline_metrics['mae'],
           baseline_metrics['r2'],
           baseline_metrics.get('mape', 0)
       ],
       'Segmented (Weighted)': [
           weighted_rmse,
           weighted_mae,
           weighted_r2,
           weighted_mape
       ],
       'Advanced Stacked Ensemble': [
           stacked_metrics['rmse'],
           stacked_metrics['mae'],
           stacked_metrics['r2'],
           stacked_metrics['mape']
       ]
   }, index=['RMSE', 'MAE', 'R2', 'MAPE (%)'])
   
   log("\nComprehensive Model Comparison:")
   log(comparison.to_string())
   
   # Calculate improvements over baseline
   baseline_r2 = baseline_metrics['r2']
   improvements = pd.DataFrame({
       'Model': ['Segmented (Weighted)', 'Advanced Stacked Ensemble'],
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
   plt.savefig('output/unified_r2_comparison.png')
   
   # RMSE comparison
   plt.figure(figsize=(10, 6))
   comparison.loc['RMSE'].plot(kind='bar')
   plt.title('RMSE Comparison')
   plt.ylabel('RMSE')
   plt.grid(axis='y', alpha=0.3)
   plt.tight_layout()
   plt.savefig('output/unified_rmse_comparison.png')
   
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
   plt.figure(figsize=(14, 8))
   sns.barplot(x='R2', y='Segment', data=segment_performance)
   plt.title('R² Score by Segment')
   plt.xlabel('R² Score')
   plt.xlim(0.7, 1.0)  # Adjusted for better visualization
   plt.grid(axis='x', alpha=0.3)
   plt.tight_layout()
   plt.savefig('output/unified_segment_r2.png')
   
   # Scatter plot of segment size vs R2
   plt.figure(figsize=(12, 8))
   sns.scatterplot(x='Size', y='R2', data=segment_performance, s=100)
   
   # Add labels for each point
   for i, row in segment_performance.iterrows():
       plt.text(row['Size'], row['R2'], row['Segment'], fontsize=8)
   
   plt.title('Segment Size vs R² Score')
   plt.xlabel('Segment Size (Number of Records)')
   plt.ylabel('R² Score')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('output/unified_segment_size_vs_r2.png')
   
   # Correlation heatmap of segment predictions
   return comparison, improvements, segment_performance

# Save models and results
def save_models_and_results(baseline_model, segment_models, stacked_model, segment_map, 
                         comparison, improvements, segment_performance):
   """Save all models and results for future use"""
   # Create models directory
   os.makedirs('output/unified_models', exist_ok=True)
   
   # Save the baseline model
   joblib.dump(baseline_model['model'], 'output/unified_models/baseline_model.pkl')
   
   # Save segment models
   for segment_name, model_info in segment_models.items():
       safe_name = segment_name.replace(" ", "_").replace("/", "_")[:50]  # Avoid filename issues
       joblib.dump(model_info['model'], f'output/unified_models/{safe_name}_model.pkl')
   
   # Save stacked model
   joblib.dump(stacked_model, 'output/unified_models/advanced_stacked_model.pkl')
   
   # Save segment map for prediction
   joblib.dump(segment_map, 'output/unified_models/segment_map.pkl')
   
   # Save metadata about models
   segment_metadata = {
       segment_name: {
           'features': model_info['selected_features'],
           'metrics': model_info['metrics'],
           'size': model_info['size']
       }
       for segment_name, model_info in segment_models.items()
   }
   
   joblib.dump(segment_metadata, 'output/unified_models/segment_metadata.pkl')
   
   # Save results to CSV
   comparison.to_csv('output/unified_model_comparison.csv')
   improvements.to_csv('output/unified_model_improvements.csv')
   segment_performance.to_csv('output/unified_segment_performance.csv')
   
   log("\nAll models and results saved to output/unified_models directory")

# Prediction function for end users
def predict_settlement(new_data, models_dir='output/unified_models'):
   """Function to make predictions on new data using the saved models"""
   # Load necessary components
   baseline_model = joblib.load(f'{models_dir}/baseline_model.pkl')
   stacked_model = joblib.load(f'{models_dir}/advanced_stacked_model.pkl')
   segment_map = joblib.load(f'{models_dir}/segment_map.pkl')
   segment_metadata = joblib.load(f'{models_dir}/segment_metadata.pkl')
   
   # Process new data similar to training data
   # Here we would need to implement the same preprocessing and feature engineering
   # For simplicity, we assume new_data is already preprocessed
   
   # Make prediction with baseline model
   baseline_pred = baseline_model.predict(new_data)
   
   # Determine which segment each sample belongs to
   # This would require implementing the segmentation logic
   # For simplicity, we'll use the baseline prediction for all
   
   return baseline_pred

# Main function
def main():
   start_time = time.time()
   
   try:
       log("======= STARTING UNIFIED SEGMENTED MODEL BUILDING PROCESS =======")
       log(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
       
       # Load data
       df = load_data('Synthetic_Data_For_Students.csv')
       
       # Preprocess data
       df = preprocess_data(df)
       
       # Engineer features
       df = engineer_features(df)
       
       # Build baseline model
       baseline_model = build_baseline_model(df)
       
       # Create optimized segmentation
       segments, segment_map = create_optimized_segments(df, ['Exceptional_Circumstances', 'AccidentType'])
       
       # Build models for all segments (tune = True, feature selection = True)
       segment_models = build_all_segment_models(segments, tune_hyperparams=True, select_features=True)
       
       # Create enhanced meta-features for stacking
       meta_features, segment_ids = create_enhanced_meta_features(df, segment_map, segment_models)
       
       # Build advanced stacked ensemble model
       stacked_model, stacked_metrics = build_advanced_stacked_ensemble(meta_features, df['SettlementValue'], segment_ids)
       
       # Comprehensive comparison and visualization
       comparison, improvements, segment_performance = compare_models(
           baseline_model['metrics'], 
           segment_models, 
           stacked_metrics
       )
       
       # Save all models and results
       save_models_and_results(
           baseline_model,
           segment_models, 
           stacked_model,
           segment_map,
           comparison, 
           improvements, 
           segment_performance
       )
       
       # Print conclusion
       log("\n" + "="*50)
       log("UNIFIED SEGMENTED MODEL BUILDING COMPLETE")
       log("="*50)
       
       # Find best model
       best_model = improvements.iloc[improvements['R2 Score'].idxmax()]['Model']
       best_r2 = improvements.iloc[improvements['R2 Score'].idxmax()]['R2 Score']
       best_improvement = improvements.iloc[improvements['R2 Score'].idxmax()]['Improvement (%)']
       
       log(f"\nBest model: {best_model}")
       log(f"R² Score: {best_r2:.4f}")
       log(f"Improvement over baseline: {best_improvement:.2f}%")
       
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
       log("\n======= UNIFIED SEGMENTED MODEL BUILDING PROCESS FINISHED =======")
       log_file.close()

if __name__ == "__main__":
   main()