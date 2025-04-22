from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Empresa(db.Model):
    __tablename__ = 'CursosTerminados'
    id = db.Column(db.Integer, primary_key=True)
    Tipo = db.Column(db.String(100), nullable=False)
    fecha_registro = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    microempresario_id = db.Column(db.Integer, db.ForeignKey('microempresarios.id'), nullable=False)