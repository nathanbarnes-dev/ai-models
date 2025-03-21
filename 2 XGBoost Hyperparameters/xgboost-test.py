import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
def load_data(file_path):
    """Load the dataset from CSV file"""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Preprocess data
def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Handle missing values in the target variable
    print("\n--- Handling Missing Values ---")
    missing_in_target = df['SettlementValue'].isna().sum()
    print(f"Missing values in SettlementValue: {missing_in_target}")
    
    # Remove rows with missing target values
    if missing_in_target > 0:
        print(f"Removing {missing_in_target} rows with missing SettlementValue")
        df = df.dropna(subset=['SettlementValue'])
    
    # Get column types
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    
    print(f"\nNumber of categorical features: {len(categorical_cols)}")
    print(f"Number of numerical features: {len(numerical_cols)}")
    
    # Add engineered features
    df = add_engineered_features(df)
    
    # Split features and target
    X = df.drop('SettlementValue', axis=1)
    y = df['SettlementValue']
    
    # Define columns by type after feature engineering
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    numerical_features = [col for col in X.columns if X[col].dtype != 'object']
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

# Feature engineering function
def add_engineered_features(df):
    """Add engineered features to improve model performance"""
    print("\n--- Adding Engineered Features ---")
    
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
            print("Could not calculate days between dates")
    
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
    
    # Create categorical flags based on string columns
    injury_flag_created = False
    for col in df_new.select_dtypes(include=['object']).columns:
        if 'Injury' in col or 'injury' in col:
            # Check if 'severe' or 'major' appears in the text
            if not injury_flag_created:
                df_new['Severe_Injury_Flag'] = df_new[col].str.lower().str.contains('severe|major|critical', na=False).astype(int)
                injury_flag_created = True
                feature_count += 1
    
    print(f"Added {feature_count} engineered features")
    return df_new

# Hyperparameter tuning for XGBoost
def tune_xgboost(X_train, y_train, X_test, y_test, preprocessor):
    """Tune XGBoost hyperparameters and evaluate performance"""
    # Create initial XGBoost pipeline
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__min_child_weight': [1, 3, 5],
        'model__n_estimators': [100, 200, 300],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__gamma': [0, 0.1, 0.2]
    }
    
    # Small grid for faster initial testing (comment out for final tuning)
    small_param_grid = {
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5],
        'model__min_child_weight': [1, 3],
        'model__n_estimators': [100, 200]
    }
    
    print("\n--- Starting XGBoost Hyperparameter Tuning ---")
    print("This may take some time...")
    
    # Create cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=small_param_grid,  # Use small_param_grid for faster testing, param_grid for full tuning
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Test Set Performance with Best Parameters ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Return the best model and results
    return best_model, grid_search.best_params_, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_r2': grid_search.best_score_
    }

# Compare with default XGBoost
def evaluate_default_xgboost(X_train, y_train, X_test, y_test, preprocessor):
    """Evaluate default XGBoost for comparison"""
    print("\n--- Evaluating Default XGBoost Model ---")
    
    # Create pipeline with default XGBoost
    default_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ])
    
    # Train the model
    default_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = default_pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Default XGBoost Performance ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Return the results
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Save model and results
def save_results(model, params, tuned_metrics, default_metrics):
    """Save the tuned model and results to files"""
    # Save the model
    joblib.dump(model, 'tuned_xgboost_model.pkl')
    print("\nTuned XGBoost model saved to 'tuned_xgboost_model.pkl'")
    
    # Save results to text file
    with open('xgboost_tuning_results.txt', 'w') as f:
        f.write("XGBOOST HYPERPARAMETER TUNING RESULTS\n")
        f.write("======================================\n\n")
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BEST PARAMETERS:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")
        
        f.write("MODEL PERFORMANCE COMPARISON:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<20}{'Default XGBoost':<20}{'Tuned XGBoost':<20}\n")
        f.write(f"{'MSE':<20}{default_metrics['mse']:<20.2f}{tuned_metrics['mse']:<20.2f}\n")
        f.write(f"{'RMSE':<20}{default_metrics['rmse']:<20.2f}{tuned_metrics['rmse']:<20.2f}\n")
        f.write(f"{'MAE':<20}{default_metrics['mae']:<20.2f}{tuned_metrics['mae']:<20.2f}\n")
        f.write(f"{'R² Score':<20}{default_metrics['r2']:<20.4f}{tuned_metrics['r2']:<20.4f}\n")
        if 'cv_r2' in tuned_metrics:
            f.write(f"{'CV R² Score':<20}{'N/A':<20}{tuned_metrics['cv_r2']:<20.4f}\n")
        f.write("-" * 50 + "\n\n")
        
        # Calculate improvement
        r2_improvement = (tuned_metrics['r2'] - default_metrics['r2']) / default_metrics['r2'] * 100
        rmse_improvement = (default_metrics['rmse'] - tuned_metrics['rmse']) / default_metrics['rmse'] * 100
        f.write(f"R² Score improved by {r2_improvement:.2f}%\n")
        f.write(f"RMSE improved by {rmse_improvement:.2f}%\n")
    
    print("Results saved to 'xgboost_tuning_results.txt'")

# Main function
def main():
    # Load data
    file_path = 'Synthetic_Data_For_Students.csv'
    df = load_data(file_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Evaluate default XGBoost
    default_metrics = evaluate_default_xgboost(X_train, y_train, X_test, y_test, preprocessor)
    
    # Tune XGBoost hyperparameters
    best_model, best_params, tuned_metrics = tune_xgboost(X_train, y_train, X_test, y_test, preprocessor)
    
    # Save results
    save_results(best_model, best_params, tuned_metrics, default_metrics)

if __name__ == "__main__":
    main()