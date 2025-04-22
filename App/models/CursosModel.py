from flask_sqlalchemy import SQLAlchemy
from . import db


class Curso(db.Model):
    __tablename__ = 'Cursos'
    id = db.Column(db.Integer, primary_key=True)
    nombre_curso = db.Column(db.String(200), nullable=False)
    categoria_id = db.Column(db.Integer, db.ForeignKey('categorias.id'), nullable=False)

    # El nombre correcto del modelo es 'CursosTerminados'
    terminados = db.relationship('CursosTerminados', backref='curso', lazy=True)
