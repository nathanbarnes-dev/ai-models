import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

# Import your model class
from model import UnifiedSegmentedModel

def evaluate_model():
    """
    Train and evaluate the unified segmented model on a test set
    """
    print("Starting model evaluation")
    start_time = time.time()
    
    # Load the dataset
    try:
        df = pd.read_csv('Synthetic_Data_For_Students.csv')
        print(f"Loaded dataset with {df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Create and train model with minimal output
    model = UnifiedSegmentedModel(verbose=False)
    print("Training model...")
    model.fit(train_df)
    
    # Evaluate on test set
    print("Evaluating model...")
    y_test = test_df['SettlementValue'].values
    y_pred = model.predict(test_df)
    
    # Calculate metrics
    r2 = r2_score(y_test[:len(y_pred)], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test[:len(y_pred)], y_pred))
    mae = mean_absolute_error(y_test[:len(y_pred)], y_pred)
    
    # Calculate MAPE with handling for zero values
    y_true_nonzero = y_test[:len(y_pred)].copy()
    nonzero_mask = y_true_nonzero != 0
    if any(nonzero_mask):
        mape = np.mean(np.abs((y_true_nonzero[nonzero_mask] - y_pred[nonzero_mask]) / 
                           y_true_nonzero[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    # Print results
    print("\n===== MODEL PERFORMANCE RESULTS =====")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Print execution time
    print(f"\nEvaluation completed in {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    evaluate_model()