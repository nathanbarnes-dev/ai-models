import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Import the UnifiedSegmentedModel class
# Make sure model.py is in the same directory as this script
try:
    from model import UnifiedSegmentedModel
    print("Successfully imported UnifiedSegmentedModel class")
except ImportError:
    # If the import fails, try to add the directory to the path
    model_path = Path(__file__).parent / "9 final model"
    if model_path.exists() and (model_path / "model.py").exists():
        sys.path.append(str(model_path))
        try:
            from model import UnifiedSegmentedModel
            print(f"Successfully imported UnifiedSegmentedModel from {model_path}")
        except ImportError as e:
            print(f"Error importing UnifiedSegmentedModel: {e}")
            print("Please ensure model.py is in the correct location")
            sys.exit(1)
    else:
        print("Error: Could not find model.py")
        print("Please ensure it's in the same directory as this script or in a '9 final model' subdirectory")
        sys.exit(1)

def load_unified_model(model_path="unified_segmented_model.pkl"):
    """
    Load the unified segmented model from a pickle file.
    
    Args:
        model_path: Path to the model pickle file
        
    Returns:
        The loaded model or None if loading fails
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found")
        return None
    
    try:
        # Use the class method for loading
        model = UnifiedSegmentedModel.load(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Failed to load the model. Please check the model file and try again.")
        return None

def predict_with_model(model, input_data_path):
    """
    Use the loaded model to make predictions on new data.
    
    Args:
        model: The loaded model
        input_data_path: Path to the CSV file with input data
        
    Returns:
        DataFrame with input data and predictions
    """
    if model is None:
        print("No model loaded, cannot make predictions")
        return None
    
    print(f"Loading input data from: {input_data_path}")
    
    if not os.path.exists(input_data_path):
        print(f"Error: Input data file '{input_data_path}' not found")
        return None
    
    try:
        # Load the input data
        df = pd.read_csv(input_data_path)
        print(f"Loaded {len(df)} records from input data")
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(df)
        
        # Add predictions to the data
        if len(predictions) == len(df):
            df['Predicted_Settlement'] = predictions
            print("Predictions added to input data")
        else:
            print(f"Warning: Prediction length ({len(predictions)}) doesn't match input data length ({len(df)})")
            # Try to add predictions for as many rows as possible
            df['Predicted_Settlement'] = np.nan
            df.loc[:len(predictions)-1, 'Predicted_Settlement'] = predictions
            
        return df
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def evaluate_predictions(df):
    """
    Evaluate prediction accuracy using multiple metrics when true values are available.
    
    Args:
        df: DataFrame with both true values (SettlementValue) and predictions (Predicted_Settlement)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if 'SettlementValue' not in df.columns or 'Predicted_Settlement' not in df.columns:
        print("Cannot evaluate predictions: missing SettlementValue or Predicted_Settlement columns")
        return None
    
    # Drop rows with missing predictions or true values
    eval_df = df.dropna(subset=['SettlementValue', 'Predicted_Settlement'])
    
    if len(eval_df) == 0:
        print("No valid data points for evaluation after dropping missing values")
        return None
    
    # Extract true values and predictions
    y_true = eval_df['SettlementValue'].values
    y_pred = eval_df['Predicted_Settlement'].values
    
    # Calculate metrics
    metrics = {}
    
    # R² Score (coefficient of determination)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    # Handle potential division by zero
    try:
        # For MAPE, we need to filter out zeros in y_true to avoid division by zero
        non_zero_mask = y_true != 0
        if sum(non_zero_mask) > 0:
            metrics['mape'] = mean_absolute_percentage_error(
                y_true[non_zero_mask], 
                y_pred[non_zero_mask]
            ) * 100  # Convert to percentage
        else:
            metrics['mape'] = np.nan
    except:
        # Calculate manually if sklearn version doesn't have MAPE
        non_zero_mask = y_true != 0
        if sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.nan
    
    # Add some additional metrics
    
    # Percentage of predictions within 10% of true value
    within_10_percent = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.1) * 100
    metrics['within_10_percent'] = within_10_percent
    
    # Percentage of predictions within 20% of true value
    within_20_percent = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.2) * 100
    metrics['within_20_percent'] = within_20_percent
    
    # Calculate over/under prediction bias
    bias = np.mean(y_pred - y_true)
    metrics['bias'] = bias
    
    # Percentage bias (relative to average true value)
    percent_bias = (bias / np.mean(y_true)) * 100
    metrics['percent_bias'] = percent_bias
    
    return metrics

def print_evaluation_metrics(metrics):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    if not metrics:
        return
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    
    print(f"\nR² Score:                  {metrics['r2']:.4f}")
    print(f"Root Mean Squared Error:   ${metrics['rmse']:.2f}")
    print(f"Mean Absolute Error:       ${metrics['mae']:.2f}")
    
    if not np.isnan(metrics['mape']):
        print(f"Mean Absolute % Error:     {metrics['mape']:.2f}%")
    
    print(f"\nPredictions within 10%:    {metrics['within_10_percent']:.2f}%")
    print(f"Predictions within 20%:    {metrics['within_20_percent']:.2f}%")
    
    print(f"\nAverage Bias:              ${metrics['bias']:.2f}")
    print(f"Percentage Bias:           {metrics['percent_bias']:.2f}%")
    
    if metrics['percent_bias'] > 0:
        print("The model tends to OVERESTIMATE settlement values on average")
    else:
        print("The model tends to UNDERESTIMATE settlement values on average")
    
    print("\nInterpretation:")
    
    # Interpret R²
    if metrics['r2'] > 0.9:
        print("- Excellent predictive performance (R² > 0.9)")
    elif metrics['r2'] > 0.8:
        print("- Good predictive performance (R² > 0.8)")
    elif metrics['r2'] > 0.7:
        print("- Moderate predictive performance (R² > 0.7)")
    else:
        print("- Model has limited predictive power (R² < 0.7)")
    
    # Interpret MAPE
    if not np.isnan(metrics['mape']):
        if metrics['mape'] < 10:
            print("- High accuracy (MAPE < 10%)")
        elif metrics['mape'] < 20:
            print("- Good accuracy (MAPE < 20%)")
        elif metrics['mape'] < 30:
            print("- Moderate accuracy (MAPE < 30%)")
        else:
            print("- Lower accuracy (MAPE > 30%)")
    
    print("="*50)

def save_predictions(df, output_path="predictions.csv"):
    """
    Save the predictions to a CSV file.
    
    Args:
        df: DataFrame with predictions
        output_path: Path to save the output CSV file
    """
    if df is None:
        print("No predictions to save")
        return
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def main():
    # Get command line arguments
    model_path = sys.argv[1] if len(sys.argv) > 1 else "unified_segmented_model.pkl"
    input_data_path = sys.argv[2] if len(sys.argv) > 2 else "Synthetic_Data_For_Students.csv"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"
    
    # Load the model
    model = load_unified_model(model_path)
    
    # Make predictions
    if model is not None:
        result_df = predict_with_model(model, input_data_path)
        
        # Save predictions
        if result_df is not None:
            save_predictions(result_df, output_path)
            
            # Display a sample of predictions
            if 'Predicted_Settlement' in result_df.columns:
                print("\nSample predictions:")
                if 'SettlementValue' in result_df.columns:
                    sample = result_df[['SettlementValue', 'Predicted_Settlement']].head(5)
                else:
                    sample = result_df[['Predicted_Settlement']].head(5)
                print(sample)
            
            # Evaluate predictions if true values are available
            if 'SettlementValue' in result_df.columns and 'Predicted_Settlement' in result_df.columns:
                print("\nEvaluating prediction accuracy...")
                metrics = evaluate_predictions(result_df)
                if metrics:
                    print_evaluation_metrics(metrics)
                    
                    # Save metrics to a text file
                    try:
                        metrics_path = output_path.replace('.csv', '_metrics.txt')
                        with open(metrics_path, 'w') as f:
                            f.write("MODEL EVALUATION METRICS\n")
                            f.write("=======================\n\n")
                            f.write(f"R² Score: {metrics['r2']:.4f}\n")
                            f.write(f"RMSE: ${metrics['rmse']:.2f}\n")
                            f.write(f"MAE: ${metrics['mae']:.2f}\n")
                            f.write(f"MAPE: {metrics['mape']:.2f}%\n")
                            f.write(f"Predictions within 10%: {metrics['within_10_percent']:.2f}%\n")
                            f.write(f"Predictions within 20%: {metrics['within_20_percent']:.2f}%\n")
                            f.write(f"Average Bias: ${metrics['bias']:.2f}\n")
                            f.write(f"Percentage Bias: {metrics['percent_bias']:.2f}%\n")
                        print(f"Metrics saved to: {metrics_path}")
                    except Exception as e:
                        print(f"Error saving metrics: {e}")
            else:
                print("\nNo true settlement values available for evaluation")

if __name__ == "__main__":
    main()