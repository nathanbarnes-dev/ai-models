import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# 1. Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset, handling potential issues"""
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Remove rows with NaN in target variable
    df = df.dropna(subset=['SettlementValue'])
    print(f"Shape after removing NaN from target: {df.shape}")
    
    # Check for infinity or extremely large values
    df = df[np.isfinite(df['SettlementValue'])]
    print(f"Shape after removing infinity values from target: {df.shape}")
    
    # Cap extremely large values at a reasonable threshold (e.g., 99.9 percentile)
    upper_threshold = df['SettlementValue'].quantile(0.999)
    df = df[df['SettlementValue'] <= upper_threshold]
    print(f"Shape after removing extreme outliers from target: {df.shape}")
    
    # Handle dates
    try:
        df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')
        df['Claim Date'] = pd.to_datetime(df['Claim Date'], errors='coerce')
        df['Claim Delay Days'] = (df['Claim Date'] - df['Accident Date']).dt.days
        df['Claim Delay Days'] = df['Claim Delay Days'].fillna(0)
    except:
        print("Error processing date columns, skipping Claim Delay Days calculation")
    
    # Fill missing values in numeric columns with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing values in categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    return df

# 2. Identify all categorical columns
def get_categorical_columns(df):
    """Get a list of all categorical columns in the dataframe"""
    return [col for col in df.select_dtypes(include=['object']).columns]

# 3. Segment data by a categorical variable
def segment_data(df, column_name, min_segment_size=50):
    """Segment the data by the specified categorical column"""
    segments = {}
    for value in df[column_name].unique():
        segment_df = df[df[column_name] == value]
        if len(segment_df) >= min_segment_size:  # Only create segments with sufficient data
            segments[value] = segment_df
    
    # Create an 'Other' segment for remaining records
    processed_values = list(segments.keys())
    other_df = df[~df[column_name].isin(processed_values)]
    if len(other_df) >= min_segment_size:
        segments['Other'] = other_df
    
    return segments

# 4. Segment data by multiple categorical variables
def segment_data_multi(df, column_names, min_segment_size=100):
    """Segment the data by multiple categorical columns"""
    segments = {}
    
    # Get unique combinations of values
    value_combinations = df[column_names].drop_duplicates()
    
    for _, row in value_combinations.iterrows():
        # Create a filter for this combination
        filter_condition = True
        segment_name_parts = []
        
        for col in column_names:
            filter_condition = filter_condition & (df[col] == row[col])
            segment_name_parts.append(f"{col}={row[col]}")
        
        segment_name = " & ".join(segment_name_parts)
        segment_df = df[filter_condition]
        
        if len(segment_df) >= min_segment_size:
            segments[segment_name] = segment_df
    
    # Create an 'Other' segment for remaining records if needed
    all_segmented = pd.concat(segments.values())
    remaining = df.loc[~df.index.isin(all_segmented.index)]
    
    if len(remaining) >= min_segment_size:
        segments['Other'] = remaining
    
    return segments

# 5. Prepare features for modeling
def prepare_features(df):
    """Prepare features for modeling"""
    # Define numeric features (adjusting based on your dataset)
    numeric_features = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                        if col != 'SettlementValue']
    
    # Define categorical features (adjusting based on your dataset)
    categorical_features = [col for col in df.select_dtypes(include=['object']).columns]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Define target variable
    y = df['SettlementValue']
    
    # Create feature matrix (excluding target)
    X = df.drop('SettlementValue', axis=1)
    
    # Select only the features we want to use
    X = X[numeric_features + categorical_features]
    
    return X, y, preprocessor

# 6. Build and evaluate model for a segment
def build_model_for_segment(segment_df, segment_name):
    """Build and evaluate a model for a single segment"""
    print(f"Building model for segment: {segment_name} (n={len(segment_df)})")
    
    # Prepare features
    X, y, preprocessor = prepare_features(segment_df)
    
    # Ensure no NaN or infinity values in y
    valid_indices = np.isfinite(y)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42
        ))
    ])
    
    # Fit model
    try:
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return pipeline, {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }
    except Exception as e:
        print(f"  Error building model for {segment_name}: {str(e)}")
        return None, None

# 7. Build a baseline model for comparison
def build_baseline_model(df):
    """Build a baseline model on the entire dataset"""
    print(f"\nBuilding baseline model on entire dataset (n={len(df)})")
    
    return build_model_for_segment(df, "Baseline")

# 8. Build segmented models
def build_segmented_models(segments, segment_type):
    """Build models for all segments"""
    print(f"\nBuilding {segment_type} models:")
    
    segment_models = {}
    segment_scores = {}
    segment_sizes = {}
    
    for segment_value, segment_df in segments.items():
        model, scores = build_model_for_segment(segment_df, f"{segment_type}: {segment_value}")
        
        if model is not None and scores is not None:
            segment_models[segment_value] = model
            segment_scores[segment_value] = scores
            segment_sizes[segment_value] = len(segment_df)
    
    return segment_models, segment_scores, segment_sizes

# 9. Calculate weighted average performance for segmented models
def calculate_weighted_average(scores, sizes):
    """Calculate weighted average performance metrics"""
    if not scores or not sizes:
        return None
    
    total_size = sum(sizes.values())
    weighted_rmse = sum(scores[seg]['RMSE'] * sizes[seg] / total_size for seg in scores)
    weighted_mae = sum(scores[seg]['MAE'] * sizes[seg] / total_size for seg in scores)
    weighted_r2 = sum(scores[seg]['R2 Score'] * sizes[seg] / total_size for seg in scores)
    
    return {
        'RMSE': weighted_rmse,
        'MAE': weighted_mae,
        'R2 Score': weighted_r2
    }

# 10. Compare all segmentation approaches
def compare_all_segmentations(baseline_scores, all_segmentation_results):
    """Compare the performance of all segmentation approaches"""
    comparison = {'Baseline': [
        baseline_scores['RMSE'],
        baseline_scores['MAE'],
        baseline_scores['R2 Score']
    ]}
    
    # Add all segmented models to comparison
    for segment_type, (_, scores, sizes) in all_segmentation_results.items():
        weighted_metrics = calculate_weighted_average(scores, sizes)
        if weighted_metrics:
            comparison[segment_type] = [
                weighted_metrics['RMSE'],
                weighted_metrics['MAE'],
                weighted_metrics['R2 Score']
            ]
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison, index=['RMSE', 'MAE', 'R2 Score'])
    
    # Sort by R2 Score (descending)
    comparison_df = comparison_df.sort_values(by='R2 Score', axis=1, ascending=False)
    
    print("\nModel Comparison (sorted by R2 Score):")
    print(comparison_df.loc['R2 Score'])
    
    # Create visualizations
    plt.figure(figsize=(14, 8))
    comparison_df.loc['R2 Score'].plot(kind='bar')
    plt.title('R² Score Comparison Across All Segmentation Approaches')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('all_segmentations_r2_comparison.png')
    
    # Display top 5 approaches
    top_5 = comparison_df.loc['R2 Score'].sort_values(ascending=False).head(5)
    plt.figure(figsize=(10, 6))
    top_5.plot(kind='bar')
    plt.title('Top 5 Segmentation Approaches by R² Score')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_5_segmentations_r2.png')
    
    # Create RMSE comparison for top 5
    top_5_names = top_5.index
    plt.figure(figsize=(10, 6))
    comparison_df.loc['RMSE'][top_5_names].plot(kind='bar')
    plt.title('RMSE for Top 5 Segmentation Approaches')
    plt.ylabel('RMSE')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_5_segmentations_rmse.png')
    
    return comparison_df

# 11. Main function
def main():
    start_time = time.time()
    
    # Load and preprocess data
    df = load_and_preprocess_data('Synthetic_Data_For_Students.csv')
    
    # Build baseline model
    baseline_model, baseline_scores = build_baseline_model(df)
    
    # Get all categorical columns
    categorical_columns = get_categorical_columns(df)
    print(f"\nCategorical columns found: {categorical_columns}")
    
    # Dictionary to store all segmentation results
    all_segmentation_results = {}
    
    # Test each categorical variable individually
    for column in categorical_columns:
        print(f"\n{'='*50}")
        print(f"Testing segmentation by: {column}")
        print(f"{'='*50}")
        
        # Count unique values
        unique_values = df[column].nunique()
        print(f"Number of unique values: {unique_values}")
        
        # Skip columns with too many unique values
        if unique_values > 20:
            print(f"Skipping {column} due to high cardinality ({unique_values} unique values)")
            continue
        
        # Segment data
        segments = segment_data(df, column)
        
        # Check if we have enough segments
        if len(segments) <= 1:
            print(f"Skipping {column} as it didn't create meaningful segments")
            continue
        
        # Build models
        models, scores, sizes = build_segmented_models(segments, column)
        
        # Store results
        all_segmentation_results[column] = (models, scores, sizes)
    
    # Identify top performing individual segmentations
    print("\nFinding top performing individual segmentations...")
    top_performers = []
    
    for segment_type, (_, scores, sizes) in all_segmentation_results.items():
        weighted_metrics = calculate_weighted_average(scores, sizes)
        if weighted_metrics:
            top_performers.append((segment_type, weighted_metrics['R2 Score']))
    
    # Sort by R2 score
    top_performers.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 5 individual segmentations
    print("\nTop 5 individual segmentation variables:")
    for i, (segment_type, r2) in enumerate(top_performers[:5], 1):
        print(f"{i}. {segment_type}: R² = {r2:.4f}")
    
    # Test combinations of top performing variables
    if len(top_performers) >= 2:
        top_columns = [col for col, _ in top_performers[:5]]
        
        print("\nTesting combinations of top performing variables...")
        
        # Try pairs of top variables
        for col1, col2 in combinations(top_columns, 2):
            combination_name = f"{col1} + {col2}"
            print(f"\n{'='*50}")
            print(f"Testing combination: {combination_name}")
            print(f"{'='*50}")
            
            # Segment data by the combination
            segments = segment_data_multi(df, [col1, col2], min_segment_size=100)
            
            # Check if we have enough segments
            if len(segments) <= 1:
                print(f"Skipping {combination_name} as it didn't create meaningful segments")
                continue
            
            # Build models
            models, scores, sizes = build_segmented_models(segments, combination_name)
            
            # Store results
            all_segmentation_results[combination_name] = (models, scores, sizes)
    
    # Compare all approaches
    comparison = compare_all_segmentations(baseline_scores, all_segmentation_results)
    
    # Identify the best approach
    best_r2 = comparison.loc['R2 Score'].max()
    best_approach = comparison.loc['R2 Score'].idxmax()
    
    print(f"\nBest performing segmentation approach: {best_approach}")
    print(f"R² Score: {best_r2:.4f}")
    
    # Compare with baseline
    baseline_r2 = comparison.loc['R2 Score']['Baseline']
    improvement = (best_r2 - baseline_r2) / baseline_r2 * 100
    print(f"Improvement over baseline: {improvement:.2f}%")
    
    # Create a summary table for the technical report
    summary = pd.DataFrame({
        'Segmentation Approach': comparison.columns,
        'R² Score': comparison.loc['R2 Score'].values,
        'RMSE': comparison.loc['RMSE'].values,
        'MAE': comparison.loc['MAE'].values,
        'Improvement (%)': [(r2 - baseline_r2) / baseline_r2 * 100 for r2 in comparison.loc['R2 Score'].values]
    })
    
    # Sort by R2 Score
    summary = summary.sort_values('R² Score', ascending=False)
    
    # Save summary to CSV
    summary.to_csv('segmentation_results_summary.csv', index=False)
    print("\nFull results saved to 'segmentation_results_summary.csv'")
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()