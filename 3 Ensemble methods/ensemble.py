import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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

# Define individual models for ensembles
def get_base_models():
    """Define individual models that will be used in ensembles"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, min_child_weight=3, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.001, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    }
    return models

# Create and evaluate voting ensemble
def evaluate_voting_ensemble(X_train, X_test, y_train, y_test, preprocessor, base_models):
    """Create and evaluate a voting ensemble"""
    print("\n--- Creating Voting Ensemble ---")
    
    # Create preprocessed pipeline for each base model
    model_pipelines = []
    for name, model in base_models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        model_pipelines.append((name, pipeline))
    
    # Train individual models to get weights
    weights = []
    trained_pipelines = {}
    for name, pipeline in model_pipelines:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        weights.append(r2)  # Use R² as weight
        trained_pipelines[name] = pipeline
        print(f"{name} individual R² score: {r2:.4f}")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Create voting ensemble with preprocessed estimators
    # We need to extract the models from the pipelines
    estimators = [(name, pipeline.named_steps['model']) for name, pipeline in model_pipelines]
    
    # Create a new pipeline with preprocessor and voting ensemble
    voting_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('voting', VotingRegressor(estimators=estimators, weights=weights))
    ])
    
    # Train and evaluate
    voting_pipeline.fit(X_train, y_train)
    y_pred = voting_pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Voting Ensemble Performance ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return voting_pipeline, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Create and evaluate stacking ensemble
def evaluate_stacking_ensemble(X_train, X_test, y_train, y_test, preprocessor, base_models):
    """Create and evaluate a stacking ensemble"""
    print("\n--- Creating Stacking Ensemble ---")
    
    # Choose models for the first level
    level0 = [
        ('rf', base_models['Random Forest']),
        ('gb', base_models['Gradient Boosting']),
        ('xgb', base_models['XGBoost'])
    ]
    
    # Choose a model for the meta-learner (second level)
    level1 = Ridge(alpha=0.1)
    
    # Create stacking ensemble
    stacking = StackingRegressor(
        estimators=level0,
        final_estimator=level1,
        cv=5,
        n_jobs=-1
    )
    
    # Create a pipeline with preprocessing
    stacking_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stacking', stacking)
    ])
    
    # Train and evaluate
    stacking_pipeline.fit(X_train, y_train)
    y_pred = stacking_pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Stacking Ensemble Performance ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return stacking_pipeline, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Create and evaluate bagging-based ensemble
def evaluate_custom_ensemble(X_train, X_test, y_train, y_test, preprocessor, base_models):
    """Create and evaluate a custom ensemble that averages predictions"""
    print("\n--- Creating Custom Ensemble (Average Predictions) ---")
    
    # Create and train individual models
    models_to_use = ['Random Forest', 'Gradient Boosting', 'XGBoost']
    trained_models = {}
    model_predictions = {}
    
    for name in models_to_use:
        print(f"Training {name}...")
        model = base_models[name]
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        model_predictions[name] = y_pred
        
        # Calculate individual metrics
        r2 = r2_score(y_test, y_pred)
        print(f"{name} individual R² score: {r2:.4f}")
    
    # Average the predictions
    ensemble_predictions = np.mean([model_predictions[name] for name in models_to_use], axis=0)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, ensemble_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, ensemble_predictions)
    r2 = r2_score(y_test, ensemble_predictions)
    
    print("\n--- Custom Ensemble Performance ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # For a custom ensemble, we need to save each model separately and implement prediction logic
    # We'll return the trained models dictionary for now
    return trained_models, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Save model and results
def save_results(results, models):
    """Save the ensemble models and results to files"""
    # Save the models
    for model_name, model in models.items():
        joblib.dump(model, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    
    # Save results to text file
    with open('ensemble_results.txt', 'w') as f:
        f.write("ENSEMBLE METHODS COMPARISON\n")
        f.write("==========================\n\n")
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL PERFORMANCE COMPARISON:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Metric':<15}{'XGBoost':<15}{'Voting':<15}{'Stacking':<15}{'Custom':<15}\n")
        f.write(f"{'MSE':<15}{results['XGBoost']['mse']:<15.2f}{results['Voting']['mse']:<15.2f}{results['Stacking']['mse']:<15.2f}{results['Custom']['mse']:<15.2f}\n")
        f.write(f"{'RMSE':<15}{results['XGBoost']['rmse']:<15.2f}{results['Voting']['rmse']:<15.2f}{results['Stacking']['rmse']:<15.2f}{results['Custom']['rmse']:<15.2f}\n")
        f.write(f"{'MAE':<15}{results['XGBoost']['mae']:<15.2f}{results['Voting']['mae']:<15.2f}{results['Stacking']['mae']:<15.2f}{results['Custom']['mae']:<15.2f}\n")
        f.write(f"{'R² Score':<15}{results['XGBoost']['r2']:<15.4f}{results['Voting']['r2']:<15.4f}{results['Stacking']['r2']:<15.4f}{results['Custom']['r2']:<15.4f}\n")
        f.write("-" * 60 + "\n\n")
        
        # Find the best model
        best_model = max(results.keys(), key=lambda k: results[k]['r2'])
        best_r2 = results[best_model]['r2']
        best_rmse = results[best_model]['rmse']
        
        f.write(f"BEST MODEL: {best_model}\n")
        f.write(f"R² Score: {best_r2:.4f}\n")
        f.write(f"RMSE: {best_rmse:.2f}\n")
    
    print("\nResults saved to 'ensemble_results.txt'")
    print("All models saved as .pkl files")

# Main function
def main():
    # Load data
    file_path = 'Synthetic_Data_For_Students.csv'
    df = load_data(file_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Get base models
    base_models = get_base_models()
    
    # Dictionary to store results
    all_results = {}
    all_models = {}
    
    # Evaluate individual XGBoost (as baseline)
    print("\n--- Evaluating Individual XGBoost Model ---")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_models['XGBoost'])
    ])
    xgb_pipeline.fit(X_train, y_train)
    xgb_pred = xgb_pipeline.predict(X_test)
    
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    
    print(f"XGBoost MSE: {xgb_mse:.2f}")
    print(f"XGBoost RMSE: {xgb_rmse:.2f}")
    print(f"XGBoost MAE: {xgb_mae:.2f}")
    print(f"XGBoost R² Score: {xgb_r2:.4f}")
    
    all_results['XGBoost'] = {
        'mse': xgb_mse,
        'rmse': xgb_rmse,
        'mae': xgb_mae,
        'r2': xgb_r2
    }
    all_models['XGBoost'] = xgb_pipeline
    
    # Evaluate Voting Ensemble
    voting_model, voting_results = evaluate_voting_ensemble(X_train, X_test, y_train, y_test, preprocessor, base_models)
    all_results['Voting'] = voting_results
    all_models['Voting'] = voting_model
    
    # Evaluate Stacking Ensemble
    stacking_model, stacking_results = evaluate_stacking_ensemble(X_train, X_test, y_train, y_test, preprocessor, base_models)
    all_results['Stacking'] = stacking_results
    all_models['Stacking'] = stacking_model
    
    # Evaluate Custom Ensemble
    custom_models, custom_results = evaluate_custom_ensemble(X_train, X_test, y_train, y_test, preprocessor, base_models)
    all_results['Custom'] = custom_results
    all_models['Custom'] = custom_models
    
    # Save results
    save_results(all_results, all_models)

if __name__ == "__main__":
    main()
