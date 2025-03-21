import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
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

# Define base models for the stacking ensemble
def get_base_models():
    """Define base models to be used in the stacking ensemble"""
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, min_child_weight=3, random_state=42))
    ]
    return base_models

# Define meta-learners to test
def get_meta_learners():
    """Define different meta-learners to test in the stacking ensemble"""
    meta_learners = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=0.1, random_state=42),
        'Lasso': Lasso(alpha=0.01, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        'SGD Regressor': SGDRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance')
    }
    return meta_learners

# Create and evaluate stacking ensembles with different meta-learners
def evaluate_stacking_with_metalearners(X_train, X_test, y_train, y_test, preprocessor):
    """Create and evaluate stacking ensembles with different meta-learners"""
    print("\n--- Testing Different Meta-Learners for Stacking Ensemble ---")
    
    # Get base models and meta-learners
    base_models = get_base_models()
    meta_learners = get_meta_learners()
    
    # Dictionary to store results
    results = {}
    
    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate each meta-learner
    for name, meta_learner in meta_learners.items():
        print(f"\nTesting meta-learner: {name}")
        
        # Create stacking ensemble
        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=cv,
            n_jobs=-1
        )
        
        # Create pipeline with preprocessing
        stacking_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('stacking', stacking)
        ])
        
        # Train the model
        stacking_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = stacking_pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Store results
        results[name] = {
            'model': stacking_pipeline,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    return results

# Save results and best model
def save_results(results):
    """Save the results and best model"""
    # Save results to text file
    with open('stacking_metalearner_results.txt', 'w') as f:
        f.write("STACKING ENSEMBLE WITH DIFFERENT META-LEARNERS\n")
        f.write("===========================================\n\n")
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("META-LEARNER PERFORMANCE COMPARISON:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Meta-Learner':<20}{'MSE':<15}{'RMSE':<15}{'MAE':<15}{'R² Score':<15}\n")
        
        # Sort by R² score (descending)
        sorted_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)}
        
        for name, metrics in sorted_results.items():
            f.write(f"{name:<20}{metrics['mse']:<15.2f}{metrics['rmse']:<15.2f}{metrics['mae']:<15.2f}{metrics['r2']:<15.4f}\n")
        
        f.write("-" * 70 + "\n\n")
        
        # Find the best meta-learner
        best_metalearner = max(results.keys(), key=lambda k: results[k]['r2'])
        best_r2 = results[best_metalearner]['r2']
        best_rmse = results[best_metalearner]['rmse']
        
        f.write(f"BEST META-LEARNER: {best_metalearner}\n")
        f.write(f"R² Score: {best_r2:.4f}\n")
        f.write(f"RMSE: {best_rmse:.2f}\n")
    
    print("\nResults saved to 'stacking_metalearner_results.txt'")
    
    # Save the best model
    best_metalearner = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_metalearner]['model']
    
    joblib.dump(best_model, 'best_stacking_model.pkl')
    print(f"Best stacking model ({best_metalearner} as meta-learner) saved to 'best_stacking_model.pkl'")

# Main function
def main():
    # Load data
    file_path = 'Synthetic_Data_For_Students.csv'
    df = load_data(file_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Evaluate stacking ensembles with different meta-learners
    results = evaluate_stacking_with_metalearners(X_train, X_test, y_train, y_test, preprocessor)
    
    # Save results and best model
    save_results(results)

if __name__ == "__main__":
    main()