import sys
import os

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from ml_utils import generate_synthetic_data, train_tabular_models, create_dummy_cnn

def main():
    models_dir = os.path.join(project_root, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=2000)
    
    print("Training tabular models...")
    train_tabular_models(df, models_dir)
    
    print("Creating and saving CNN model...")
    create_dummy_cnn(models_dir)
    
    print("All models trained and saved.")

if __name__ == "__main__":
    main()
