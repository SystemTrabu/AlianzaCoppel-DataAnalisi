from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
class Curso(db.Model):
    __tablename__ = 'cursos'
    id = db.Column(db.Integer, primary_key=True)
    nombre_curso = db.Column(db.String(200), nullable=False)
    categoria_id = db.Column(db.Integer, db.ForeignKey('categorias.id'), nullable=False)
    # Relación con cursos terminados
    terminados = db.relationship('CursoTerminado', backref='curso', lazy=True)