import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as XGBRegressor
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
def load_data(file_path):
    """Load the dataset from CSV file"""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Exploratory Data Analysis
def explore_data(df):
    """Get column types for preprocessing"""
    # Display basic information
    print("\n--- Dataset Overview ---")
    print(f"Dataset shape: {df.shape}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_counts = missing_values[missing_values > 0]
    if len(missing_counts) > 0:
        print("\n--- Missing Values ---")
        print(missing_counts)
    
    # Get column types
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    
    print(f"\nNumber of categorical features: {len(categorical_cols)}")
    print(f"Number of numerical features: {len(numerical_cols)}")
    
    return categorical_cols, numerical_cols

# Preprocess data
def preprocess_data(df, categorical_cols, numerical_cols):
    """Preprocess the data for modeling"""
    # Handle missing values in the target variable
    print("\n--- Handling Missing Values ---")
    missing_in_target = df['SettlementValue'].isna().sum()
    print(f"Missing values in SettlementValue: {missing_in_target}")
    
    # Remove rows with missing target values
    if missing_in_target > 0:
        print(f"Removing {missing_in_target} rows with missing SettlementValue")
        df = df.dropna(subset=['SettlementValue'])
        
    # Split features and target
    X = df.drop('SettlementValue', axis=1)
    y = df['SettlementValue']
    
    # Define columns by type
    categorical_features = [col for col in categorical_cols if col in X.columns]
    numerical_features = [col for col in numerical_cols if col in X.columns]
    
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

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train different models and evaluate their performance"""
    # Dictionary to store model results
    model_results = {}
    
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=100, random_state=42)
    r3 = KNeighborsRegressor()

    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor.XGBRegressor(random_state=42),
        'Voting Regressor': VotingRegressor(estimators=[('lr', r1), ('rf', r2), ('r3', r3)]),
        'SGD Regressor': SGDRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Extra Tree Regressor': ExtraTreeRegressor()
    }
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
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
        model_results[name] = {
            'model': pipeline,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    return model_results

# Feature importance info
def get_best_model_info(model_results):
    """Get best model information"""
    # Find the best model based on R² score
    best_model_name = max(model_results, key=lambda k: model_results[k]['r2'])
    best_model = model_results[best_model_name]
    
    print(f"\n--- Best Model: {best_model_name} ---")
    print(f"MSE: {best_model['mse']:.2f}")
    print(f"RMSE: {best_model['rmse']:.2f}")
    print(f"MAE: {best_model['mae']:.2f}")
    print(f"R² Score: {best_model['r2']:.4f}")
    
    return best_model_name, best_model

# Main function
def main():
    # Load data
    file_path = 'Synthetic_Data_For_Students.csv'
    df = load_data(file_path)
    
    # Explore data
    categorical_cols, numerical_cols = explore_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, categorical_cols, numerical_cols)
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Get best model info
    best_model_name, best_model_results = get_best_model_info(model_results)
    
    # Save the best model
    best_model = model_results[best_model_name]['model']
    
    # Save the model to disk
    model_filename = 'best_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\nBest model ({best_model_name}) saved to '{model_filename}'")
    
    # Compare model performances
    models_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'RMSE': [model_results[model]['rmse'] for model in model_results],
        'MAE': [model_results[model]['mae'] for model in model_results],
        'R² Score': [model_results[model]['r2'] for model in model_results]
    })
    
    print("\n--- Model Performance Comparison ---")
    print(models_df.sort_values('R² Score', ascending=False))
    
    # Save model results to text file
    with open('model_results.txt', 'w', encoding='utf-8') as f:
        f.write("INSURANCE SETTLEMENT PREDICTION MODEL RESULTS\n")
        f.write("===========================================\n\n")
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("MODEL PERFORMANCE METRICS:\n")
        f.write(models_df.sort_values('R² Score', ascending=False).to_string(index=False))
        f.write("\n\n")
        
        # Add best model details
        best_model_name = models_df.sort_values('R² Score', ascending=False)['Model'].iloc[0]
        best_model_r2 = models_df.sort_values('R² Score', ascending=False)['R² Score'].iloc[0]
        best_model_rmse = models_df.sort_values('R² Score', ascending=False)['RMSE'].iloc[0]
        
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"R² Score: {best_model_r2:.4f}\n")
        f.write(f"RMSE: {best_model_rmse:.2f}\n\n")
        
        f.write("MODEL DETAILS:\n")
        for name, results in model_results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  MSE: {results['mse']:.2f}\n")
            f.write(f"  RMSE: {results['rmse']:.2f}\n")
            f.write(f"  MAE: {results['mae']:.2f}\n")
            f.write(f"  R² Score: {results['r2']:.4f}\n")
    
    print(f"\nModel results saved to 'model_results.txt'")

if __name__ == "__main__":
    main()