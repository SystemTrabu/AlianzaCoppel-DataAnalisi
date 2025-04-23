from datetime import datetime
from . import db

class MicroEmpresario(db.Model):
    __tablename__ = 'microempresario' 

    id = db.Column(db.Integer, primary_key=True)
    nombre_empresario = db.Column(db.String(100), nullable=False)
    correo = db.Column(db.String(100), nullable=False)
    genero = db.Column(db.String(100), nullable=False)
    n_telefono = db.Column(db.String(100), nullable=False)
    codigo_postal = db.Column(db.Integer)
    Webinars = db.Column(db.Integer)
    fecha_registro = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    fecha_nacimiento = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    nivel_educativo= db.Column(db.String(100), nullable=False)
    colaborador_id = db.Column(db.Integer, db.ForeignKey('colaborador.id'), nullable=False)
    empresa_id = db.Column(db.Integer, db.ForeignKey('empresas.id'), nullable=False)
    cursos_terminados = db.relationship('CursosTerminados', backref='microempresario', lazy=True)
