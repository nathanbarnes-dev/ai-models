import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Create output directory
os.makedirs('output', exist_ok=True)

# Create a log file
log_filename = f'output/model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
log_file = open(log_filename, 'w')

# Function to log to both console and file
def log(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()  # Make sure it's written immediately

# 1. Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset, handling potential issues"""
    # Load data
    df = pd.read_csv(filepath)
    
    log(f"Original dataset shape: {df.shape}")
    
    # Remove rows with NaN in target variable
    df = df.dropna(subset=['SettlementValue'])
    log(f"Shape after removing NaN from target: {df.shape}")
    
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

# 2. Segment data based on Exceptional_Circumstances
def create_segments(df, column='Exceptional_Circumstances', min_segment_size=50):
    """Create segments based on specified column"""
    segments = {}
    
    for value in df[column].unique():
        segment_df = df[df[column] == value]
        if len(segment_df) >= min_segment_size:
            segments[value] = segment_df
    
    # Create an 'Other' segment for remaining records
    processed_values = list(segments.keys())
    other_df = df[~df[column].isin(processed_values)]
    if len(other_df) >= min_segment_size:
        segments['Other'] = other_df
    
    return segments

# 3. Prepare features for modeling with scikit-learn pipeline
def get_xy_data(df):
    """Extract X and y from dataframe with minimal preprocessing"""
    # Define target variable
    y = df['SettlementValue']
    
    # Create feature matrix (excluding target)
    X = df.drop('SettlementValue', axis=1)
    
    return X, y

# 4. Build model for a segment
def build_model_for_segment(segment_df, segment_name):
    """Build a model for a single segment"""
    log(f"Building model for segment: {segment_name} (n={len(segment_df)})")
    
    # Get numeric and categorical columns
    numeric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    # Create a copy of the data with only numeric columns
    df_numeric = segment_df[numeric_cols + ['SettlementValue']].copy()
    
    # Get X and y
    X, y = get_xy_data(df_numeric)
    
    # Ensure no NaN or infinity values in y
    valid_indices = np.isfinite(y)
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use basic XGBoost model
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=42
    )
    
    # Fit model
    try:
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        # Plot feature importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title(f'Feature Importance for {segment_name}')
            plt.tight_layout()
            plt.savefig(f'output/feature_importance_{segment_name.replace(" ", "_")}.png')
            
            # Save top 10 feature importances to log
            log(f"\nTop 10 feature importances for {segment_name}:")
            for idx, row in feature_importance.head(10).iterrows():
                log(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        return model, {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }
    except Exception as e:
        log(f"  Error building model for {segment_name}: {str(e)}")
        return None, None

# 5. Build baseline model
def build_baseline_model(df):
    """Build a baseline model on the entire dataset"""
    log(f"\nBuilding baseline model on entire dataset (n={len(df)})")
    
    return build_model_for_segment(df, "Baseline")

# 6. Build segmented models
def build_segmented_models(segments):
    """Build models for all segments"""
    log(f"\nBuilding segmented models:")
    
    segment_models = {}
    segment_scores = {}
    segment_sizes = {}
    
    for segment_value, segment_df in segments.items():
        model, scores = build_model_for_segment(segment_df, f"Segment_{segment_value}")
        
        if model is not None and scores is not None:
            segment_models[segment_value] = model
            segment_scores[segment_value] = scores
            segment_sizes[segment_value] = len(segment_df)
    
    return segment_models, segment_scores, segment_sizes

# 7. Create stacked model
def build_stacked_model(df, segments, segment_models):
    """Build a stacked model that combines segment predictions with a meta-model"""
    log("\nBuilding stacked ensemble model")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'SettlementValue']
    
    # Create a copy of the data with only numeric columns
    df_numeric = df[numeric_cols + ['SettlementValue']].copy()
    
    # Get the segment column
    segmentation_col = 'Exceptional_Circumstances'
    
    # Add segment as a feature
    df_numeric['segment'] = df[segmentation_col].apply(
        lambda x: x if x in segment_models else 'Other'
    )
    
    # Create segment prediction features
    df_numeric['segment_prediction'] = 0
    
    # Get segment predictions
    for segment, model in segment_models.items():
        # Get segment-specific data
        mask = df_numeric['segment'] == segment
        if not any(mask):
            continue
        
        segment_data = df_numeric.loc[mask, numeric_cols]
        
        # Get predictions
        try:
            preds = model.predict(segment_data)
            df_numeric.loc[mask, 'segment_prediction'] = preds
        except Exception as e:
            log(f"  Error getting predictions for segment {segment}: {str(e)}")
    
    # Drop segment column
    df_numeric = df_numeric.drop('segment', axis=1)
    
    # Get X and y for stacked model
    X = df_numeric.drop('SettlementValue', axis=1)
    y = df_numeric['SettlementValue']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create stacked model
    stacked_model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
    
    # Fit model
    stacked_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = stacked_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    log(f"Stacked model performance:")
    log(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': stacked_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Stacked Model')
    plt.tight_layout()
    plt.savefig('output/feature_importance_stacked_model.png')
    
    # Save feature importances to log
    log("\nFeature importances for stacked model:")
    for idx, row in feature_importance.iterrows():
        log(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return stacked_model, {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

# 8. Compare models
def compare_models(baseline_scores, segmented_scores, stacked_scores, segment_sizes):
    """Compare model performances and create visualizations"""
    # Calculate weighted average for segmented models
    total_size = sum(segment_sizes.values())
    weighted_rmse = sum(segmented_scores[seg]['RMSE'] * segment_sizes[seg] / total_size for seg in segmented_scores)
    weighted_mae = sum(segmented_scores[seg]['MAE'] * segment_sizes[seg] / total_size for seg in segmented_scores)
    weighted_r2 = sum(segmented_scores[seg]['R2 Score'] * segment_sizes[seg] / total_size for seg in segmented_scores)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Baseline': [
            baseline_scores['RMSE'],
            baseline_scores['MAE'],
            baseline_scores['R2 Score']
        ],
        'Segmented Models (Weighted)': [
            weighted_rmse,
            weighted_mae,
            weighted_r2
        ],
        'Stacked Model': [
            stacked_scores['RMSE'],
            stacked_scores['MAE'],
            stacked_scores['R2 Score']
        ]
    }, index=['RMSE', 'MAE', 'R2 Score'])
    
    log("\nModel Comparison:")
    log(comparison.to_string())
    
    # Calculate improvements
    baseline_r2 = baseline_scores['R2 Score']
    improvements = pd.DataFrame({
        'Model': ['Segmented Models', 'Stacked Model'],
        'R2 Score': [weighted_r2, stacked_scores['R2 Score']],
        'Improvement (%)': [
            (weighted_r2 - baseline_r2) / baseline_r2 * 100,
            (stacked_scores['R2 Score'] - baseline_r2) / baseline_r2 * 100
        ]
    })
    
    log("\nImprovements over Baseline:")
    log(improvements.to_string())
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    comparison.loc['R2 Score'].plot(kind='bar')
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/r2_comparison.png')
    
    plt.figure(figsize=(10, 6))
    comparison.loc['RMSE'].plot(kind='bar')
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/rmse_comparison.png')
    
    # Create improvement visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Improvement (%)', data=improvements)
    plt.title('Improvement Over Baseline (%)')
    plt.ylabel('Improvement (%)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/improvement_comparison.png')
    
    # Plot individual segment performances
    plt.figure(figsize=(12, 8))
    segment_r2 = {k: v['R2 Score'] for k, v in segmented_scores.items()}
    segment_sizes_series = pd.Series(segment_sizes)
    
    # Create a DataFrame with both R2 and segment size
    segment_performance = pd.DataFrame({
        'R2 Score': pd.Series(segment_r2),
        'Segment Size': segment_sizes_series
    }).sort_values('R2 Score', ascending=False)
    
    # Log segment performance
    log("\nPerformance by segment:")
    log(segment_performance.to_string())
    
    # Plot R2 scores
    segment_performance['R2 Score'].plot(kind='bar')
    plt.title('R² Score by Segment')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/segment_r2_comparison.png')
    
    return comparison, improvements

# 9. Main function
def main():
    try:
        log("======= STARTING MODEL BUILDING PROCESS =======")
        log(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load and preprocess data
        df = load_and_preprocess_data('Synthetic_Data_For_Students.csv')
        
        # Log data information
        log("\nDataset Information:")
        log(f"  Number of records: {len(df)}")
        log(f"  Number of features: {len(df.columns) - 1}")  # Excluding target
        log(f"  Numeric features: {len(df.select_dtypes(include=['float64', 'int64']).columns)}")
        log(f"  Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
        
        # Log target variable statistics
        log("\nTarget Variable (SettlementValue) Statistics:")
        log(f"  Mean: {df['SettlementValue'].mean():.2f}")
        log(f"  Median: {df['SettlementValue'].median():.2f}")
        log(f"  Min: {df['SettlementValue'].min():.2f}")
        log(f"  Max: {df['SettlementValue'].max():.2f}")
        log(f"  Standard Deviation: {df['SettlementValue'].std():.2f}")
        
        # Build baseline model with numeric features only
        baseline_model, baseline_scores = build_baseline_model(df)
        
        # Segment the data based on Exceptional_Circumstances
        segments = create_segments(df, 'Exceptional_Circumstances')
        
        # Log segmentation information
        log("\nSegmentation Information:")
        log(f"  Segmentation variable: Exceptional_Circumstances")
        log(f"  Number of segments: {len(segments)}")
        for segment, segment_df in segments.items():
            log(f"  Segment '{segment}': {len(segment_df)} records ({len(segment_df)/len(df)*100:.1f}%)")
        
        # Build segmented models
        segment_models, segment_scores, segment_sizes = build_segmented_models(segments)
        
        # Build stacked model
        stacked_model, stacked_scores = build_stacked_model(df, segments, segment_models)
        
        # Compare all models
        comparison, improvements = compare_models(
            baseline_scores, 
            segment_scores, 
            stacked_scores, 
            segment_sizes
        )
        
        # Save summary results to CSV
        comparison.to_csv('output/model_comparison_summary.csv')
        improvements.to_csv('output/model_improvements_summary.csv')
        
        # Print final summary
        log("\n" + "="*50)
        log("MODEL BUILDING COMPLETE")
        log("="*50)
        log(f"Results saved in the 'output' directory")
        log(f"Detailed log saved to: {log_filename}")
        
        # Find best model
        best_model = improvements.loc[improvements['R2 Score'].idxmax(), 'Model']
        best_r2 = improvements.loc[improvements['R2 Score'].idxmax(), 'R2 Score']
        best_improvement = improvements.loc[improvements['R2 Score'].idxmax(), 'Improvement (%)']
        
        log("\nBest model performance:")
        log(f"  Model: {best_model}")
        log(f"  R² Score: {best_r2:.4f}")
        log(f"  Improvement: {best_improvement:.2f}%")
        
        # Add conclusion and recommendations
        log("\nConclusion and Recommendations:")
        if best_improvement > 5:
            log("  The segmented modeling approach shows substantial improvement over the baseline model.")
            log("  Recommendation: Implement the stacked model approach in production.")
        elif best_improvement > 2:
            log("  The segmented modeling approach shows moderate improvement over the baseline model.")
            log("  Recommendation: Consider implementing the segmented approach, particularly for critical segments.")
        else:
            log("  The segmented modeling approach shows minimal improvement over the baseline model.")
            log("  Recommendation: Further investigate additional segmentation variables or feature engineering approaches.")
        
    except Exception as e:
        log(f"\nERROR in main execution: {str(e)}")
    finally:
        # Close the log file
        log("\n======= MODEL BUILDING PROCESS FINISHED =======")
        log_file.close()

if __name__ == "__main__":
    main()