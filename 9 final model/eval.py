import pandas as pd
import numpy as np
import os
import sys
import joblib
from pathlib import Path

# First, let's make sure we can find the model module
# Add the directory containing model.py to the Python path if needed
current_dir = Path(__file__).parent
if not (current_dir / "model.py").exists():
    # Check if the module is in a subdirectory
    if (current_dir / "9 final model" / "model.py").exists():
        sys.path.append(str(current_dir / "9 final model"))
    else:
        # Look for model.py in any subdirectory
        model_paths = list(current_dir.glob("**/model.py"))
        if model_paths:
            sys.path.append(str(model_paths[0].parent))

# Now try to import the UnifiedSegmentedModel class
try:
    from model import UnifiedSegmentedModel
    print("Successfully imported UnifiedSegmentedModel class")
except ImportError as e:
    print(f"Error importing UnifiedSegmentedModel: {e}")
    print("Please ensure model.py is in the current directory or a subdirectory")
    sys.exit(1)

def load_model(model_path="unified_segmented_model.pkl"):
    """
    Load the trained UnifiedSegmentedModel from a pickle file.
    
    Args:
        model_path: Path to the saved model pickle file
    
    Returns:
        The loaded model
    """
    print(f"Attempting to load model from: {model_path}")
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        
        # Look for the model file in different locations
        possible_paths = [
            Path(model_path),
            Path.cwd() / model_path,
            Path.cwd() / "9 final model" / model_path,
            Path.cwd() / "output" / model_path,
            Path.cwd() / "output" / "unified_models" / model_path
        ]
        
        found = False
        for path in possible_paths:
            if path.exists():
                print(f"Found model at: {path}")
                model_path = str(path)
                found = True
                break
                
        if not found:
            # Look for any .pkl files
            pkl_files = list(Path.cwd().glob("**/*.pkl"))
            if pkl_files:
                print(f"Found these pickle files that might be the model:")
                for i, path in enumerate(pkl_files):
                    print(f"  {i+1}. {path}")
                print("Please specify the correct path when calling this script")
            
            print("Exiting...")
            return None
    
    try:
        # Try to load with joblib directly
        model = joblib.load(model_path)
        print("Model loaded successfully using joblib")
        
        # Check if it's the correct type
        if not isinstance(model, UnifiedSegmentedModel):
            print(f"Warning: Loaded object is not a UnifiedSegmentedModel, it's a {type(model)}")
            
            # If it's a dictionary, it might be model components rather than the full model
            if isinstance(model, dict) and 'model' in model:
                print("Loaded object appears to be a model component dictionary")
                return model
                
        return model
        
    except Exception as e:
        print(f"Error loading model with joblib: {e}")
        
        # Try using the class method instead
        try:
            model = UnifiedSegmentedModel.load(model_path)
            print("Model loaded successfully using class method")
            return model
        except Exception as e2:
            print(f"Error loading model with class method: {e2}")
            return None

def test_model_predictions(model, test_data_path="Synthetic_Data_For_Students.csv"):
    """
    Test the loaded model by making predictions on test data.
    
    Args:
        model: The loaded model
        test_data_path: Path to the test data
    """
    # Make sure we have a model
    if model is None:
        print("No model to test")
        return
    
    # Load test data
    try:
        df = pd.read_csv(test_data_path)
        print(f"Loaded test data with {len(df)} rows")
    except Exception as e:
        print(f"Error loading test data: {e}")
        
        # Look for the CSV file
        csv_files = list(Path.cwd().glob("**/*.csv"))
        if csv_files:
            print(f"Found these CSV files that might contain test data:")
            for i, path in enumerate(csv_files):
                print(f"  {i+1}. {path}")
            print("Please specify the correct path when calling this script")
        return
    
    # Make predictions
    try:
        print("Making predictions with the loaded model...")
        predictions = model.predict(df)
        print(f"Successfully made predictions for {len(predictions)} records")
        print(f"First 5 predictions: {predictions[:5]}")
        
        # If the data has the target variable, evaluate the model
        if 'SettlementValue' in df.columns:
            try:
                r2 = model.score(df)
                print(f"R² Score on loaded data: {r2:.4f}")
            except Exception as e:
                print(f"Error calculating R² score: {e}")
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        print("This suggests the model might not be correctly loaded or initialized")

if __name__ == "__main__":
    # Get model path from command line if provided
    model_path = sys.argv[1] if len(sys.argv) > 1 else "unified_segmented_model.pkl"
    
    # Load the model
    model = load_model(model_path)
    
    # Test the model if it was loaded successfully
    if model is not None:
        # Get test data path from command line if provided
        test_data_path = sys.argv[2] if len(sys.argv) > 2 else "Synthetic_Data_For_Students.csv"
        test_model_predictions(model, test_data_path)