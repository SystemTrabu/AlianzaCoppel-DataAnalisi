from datetime import datetime
from . import db

class Empresa(db.Model):
    __tablename__ = 'empresas'
    id = db.Column(db.Integer, primary_key=True)
    nombre_empresa = db.Column(db.String(100), nullable=False)
    tipo_empresa = db.Column(db.String(100), nullable=False)
    nivel_madurez = db.Column(db.String(100), nullable=False)
    n_empleados  = db.Column(db.Integer, nullable=False)
    negocio_familiar  = db.Column(db.Boolean, default=True, nullable=False)
    ingresos_semanales =  db.Column(db.Integer, nullable=False)
    microempresarios = db.relationship('MicroEmpresario', backref='empresa', lazy=True)
    antiguedad=  db.Column(db.Integer, nullable=False)
    def __init__(self, nombre_empresa, tipo_empresa, nivel_madurez, n_empleados, negocio_familiar, ingresos_semanales):
        self.nombre_empresa = nombre_empresa
        self.tipo_empresa = tipo_empresa
        self.nivel_madurez = nivel_madurez
        self.n_empleados = n_empleados
        self.negocio_familiar = negocio_familiar
        self.ingresos_semanales = ingresos_semanales

    def __repr__(self):
        return f'Empresa({self.nombre_empresa}, {self.tipo_empresa}, {self.nivel_madurez}, {self.n_empleados}, {self.negocio_familiar}, {self.ingresos_semanales})'