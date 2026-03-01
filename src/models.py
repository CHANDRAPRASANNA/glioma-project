from flask_login import UserMixin
from . import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class PatientData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Optional for now
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    tumor_grade = db.Column(db.String(50), nullable=False)
    symptoms = db.Column(db.String(200))
    idh_mutation = db.Column(db.Integer) # 0 or 1
    mgmt_methylation = db.Column(db.Integer) # 0 or 1
    codeletion_1p19q = db.Column(db.Integer) # 0 or 1
    mri_path = db.Column(db.String(200))
    prediction_result = db.Column(db.String(50))
    prediction_confidence = db.Column(db.Float)
    shap_plot = db.Column(db.Text) # Base64 string
    lime_plot = db.Column(db.Text) # Base64 string
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
