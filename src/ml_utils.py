import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2


def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 90, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Tumor_Grade': np.random.choice(['Grade II', 'Grade III', 'Grade IV'], n_samples),
        'IDH_Mutation': np.random.randint(0, 2, n_samples),
        'MGMT_Methylation': np.random.randint(0, 2, n_samples),
        '1p19q_Codeletion': np.random.randint(0, 2, n_samples),
        # Synthetic MRI feature (simulated for tabular model integration)
        'MRI_Tumor_Size': np.random.uniform(1.0, 10.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Logic for target "Risk": High risk if Grade IV or Old + No Mutation
    def get_risk(row):
        score = 0
        if row['Tumor_Grade'] == 'Grade IV': score += 5
        if row['Tumor_Grade'] == 'Grade III': score += 3
        if row['Age'] > 60: score += 2
        if row['IDH_Mutation'] == 0: score += 2 # Wildtype is worse
        if row['MRI_Tumor_Size'] > 5.0: score += 1
        
        return 1 if score > 4 else 0 # 1 = High Risk, 0 = Low Risk
        
    df['Target'] = df.apply(get_risk, axis=1)
    return df

def train_tabular_models(df, models_dir):
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Preprocessing
    numeric_features = ['Age', 'MRI_Tumor_Size']
    categorical_features = ['Gender', 'Tumor_Grade']
    # Binary features like IDH don't strictly need OHE but standardization helps SVM
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', LabelEncoder()) # Pipeline with LabelEncoder is tricky, using OneHot usually better
    ])
    
    # Using simple pandas get_dummies for simplicity in this script or ColumnTransformer with OneHot
    # Let's use ColumnTransformer for robustness
    from sklearn.preprocessing import OneHotEncoder
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('passthrough', 'passthrough', ['IDH_Mutation', 'MGMT_Methylation', '1p19q_Codeletion'])
        ])

    # Models
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = SVC(probability=True, random_state=1)
    clf4 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)

    # Voting Classifier
    eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('svm', clf3), ('xgb', clf4)], voting='soft')

    # Create a full pipeline including preprocessing for the final model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', eclf1)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Voting Classifier Accuracy: {score:.4f}")
    
    # Save model
    with open(os.path.join(models_dir, 'voting_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    return model

def create_dummy_cnn(models_dir):
    # Create a simple CNN that accepts 128x128x3 images
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary classification output
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Save the model architecture and weights
    model.save(os.path.join(models_dir, 'cnn_model_v2.h5'))
    print("Dummy CNN model saved to cnn_model_v2.h5")

def predict_cnn_model(model_path, image_path):
    """
    Loads the CNN model and predicts if the image contains a tumor.
    Returns likelihood (0-1) and label.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return 0.0, "Model Missing"
        
    try:
        model = models.load_model(model_path)
        
        # Preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return 0.0, "Image Error"
            
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0) # Add batch dimension
        
        prediction = model.predict(img)[0][0]
        
        # Output is sigmoid (0-1), >0.5 is tumor
        label = "Tumor Detected" if prediction > 0.5 else "No Tumor"
        return float(prediction), label
        
    except Exception as e:
        print(f"Error in CNN prediction: {e}")
        return 0.0, "Error"
