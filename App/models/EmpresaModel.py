from datetime import datetime
from . import db


class Empresa(db.Model):
    __tablename__ = 'empresas'
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    tipo = db.Column(db.String(100), nullable=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    microempresarios = db.relationship('Microempresario', backref='empresa', lazy=True)