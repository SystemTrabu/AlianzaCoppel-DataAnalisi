from flask_sqlalchemy import SQLAlchemy
from . import db


class Usuario(db.Model):
    __tablename__ = 'colaborador'
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    N_Empleado=db.Column(db.Integer)
    microempresarios = db.relationship('MicroEmpresario', backref='colaborador', lazy=True)