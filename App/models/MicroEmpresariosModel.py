# models/area.py
from . import db

class MicroEmpresario(db.Model):
    __tablename__ = 'microempresario'  # correcto en singular

    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    N_telefono = db.Column(db.String(100), nullable=False)
    CodigoPostal = db.Column(db.Integer)
    Webinars = db.Column(db.Integer)
    colaborador_id = db.Column(db.Integer, db.ForeignKey('colaborador.id'), nullable=False)
    empresa_id = db.Column(db.Integer, db.ForeignKey('empresas.id'), nullable=False)

    cursos_terminados = db.relationship('CursosTerminados', backref='microempresario', lazy=True)
