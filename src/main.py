import os
from flask import Blueprint, render_template, request, redirect, url_for, current_app, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from .models import PatientData
from . import db
from .ml_utils import train_tabular_models, predict_cnn_model # In real app we load, not train
from .xai_utils import generate_shap_plot, generate_lime_explanation, generate_correlation_matrix, generate_cnn_accuracy_curve
import pickle
import pandas as pd
import numpy as np

# Note: In production we would move these imports and loading to a proper singleton or lazy loader
# For now we will load models at module level or request level carefully

main = Blueprint('main', __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'voting_model.pkl')
CNN_PATH = os.path.join(BASE_DIR, 'models', 'cnn_model_v2.h5')

print(f"--- DEBUG: Looking for model at {MODEL_PATH} ---")

def get_model():
    if not os.path.exists(MODEL_PATH):
        print(f"--- ERROR: Model file not found at {MODEL_PATH} ---")
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    patients = PatientData.query.filter_by(user_id=current_user.id).order_by(PatientData.timestamp.desc()).all()
    return render_template('dashboard.html', patients=patients)

@main.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Extract Form Data
            age = int(request.form.get('age'))
            gender = request.form.get('gender')
            tumor_grade = request.form.get('tumor_grade')
            symptoms = request.form.get('symptoms')
            idh = int(request.form.get('idh_mutation'))
            mgmt = int(request.form.get('mgmt_methylation'))
            codeletion = int(request.form.get('codeletion_1p19q'))
            
            # MRI Handling
            mri_file = request.files.get('mri_scan')
            mri_filename = None
            mri_size_feature = 3.0 # Default dummy feature
            
            if mri_file and mri_file.filename != '':
                filename = secure_filename(mri_file.filename)
                save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                mri_file.save(save_path)
                mri_filename = filename
                
                # Use the trained CNN model
                print(f"--- DEBUG: Predicting on image {save_path} ---")
                cnn_prob, cnn_label = predict_cnn_model(CNN_PATH, save_path)
                print(f"--- DEBUG: CNN Result: {cnn_prob:.4f} ({cnn_label}) ---")
                
                # Map CNN result to a feature for the tabular model (optional, but keeps the flow)
                # If tumor detected (prob > 0.5), we can assume a larger size or risk factor
                if cnn_prob > 0.5:
                        mri_size_feature = 5.0 + (cnn_prob * 5.0) # Map 0.5-1.0 to 7.5-10.0
                else:
                        mri_size_feature = cnn_prob * 5.0 # Map 0.0-0.5 to 0.0-2.5


            # Prepare Data for Prediction
            input_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Tumor_Grade': tumor_grade,
                'IDH_Mutation': idh,
                'MGMT_Methylation': mgmt,
                '1p19q_Codeletion': codeletion,
                'MRI_Tumor_Size': mri_size_feature
            }])
            
            print("--- DEBUG: Prediction Input ---")
            print(input_data)
            print("-------------------------------")
            
            # Predict
            model = get_model()
            if model:
                # The model pipeline handles preprocessing
                prediction = model.predict(input_data)[0]
                proba_arr = model.predict_proba(input_data)[0]
                proba = proba_arr[1]
                
                print(f"--- DEBUG: Prediction: {prediction}, Proba: {proba_arr} ---")
                
                result_str = "High Risk" if prediction == 1 else "Low Risk"
                confidence = round(proba * 100, 2)
                
                # Generate XAI Plots
                shap_plot = generate_shap_plot(model, input_data)
                lime_plot = generate_lime_explanation(model, input_data)
                
            else:
                result_str = "Error: Model not found"
                confidence = 0.0
                shap_plot = None
                lime_plot = None

            # Save to DB
            new_patient = PatientData(
                user_id=current_user.id,
                age=age,
                gender=gender,
                tumor_grade=tumor_grade,
                symptoms=symptoms,
                idh_mutation=idh,
                mgmt_methylation=mgmt,
                codeletion_1p19q=codeletion,
                mri_path=mri_filename,
                prediction_result=result_str,
                prediction_confidence=confidence,
                shap_plot=shap_plot,
                lime_plot=lime_plot
            )
            db.session.add(new_patient)
            db.session.commit()

            return redirect(url_for('main.result', patient_id=new_patient.id))
            
        except Exception as e:
            flash(f"Error during prediction: {str(e)}")
            return redirect(url_for('main.predict'))

    return render_template('predict.html')

@main.route('/result/<int:patient_id>')
@login_required
def result(patient_id):
    patient = PatientData.query.get_or_404(patient_id)
    if patient.user_id != current_user.id:
        flash("Access denied.")
        return redirect(url_for('main.dashboard'))
        
    corr_plot = generate_correlation_matrix()
    cnn_acc_plot = generate_cnn_accuracy_curve()
        
    return render_template('result.html', patient=patient, corr_plot=corr_plot, cnn_acc_plot=cnn_acc_plot)

@main.route('/about')
def about():
    return render_template('about.html')
