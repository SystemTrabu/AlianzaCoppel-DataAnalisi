from flask_sqlalchemy import SQLAlchemy
from . import db


class Categoria(db.Model):
    __tablename__ = 'categorias'
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    # Relación con cursos
    cursos = db.relationship('Curso', backref='categoria', lazy=True)