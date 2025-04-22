from datetime import datetime
from . import db

class CursosTerminados(db.Model):
    __tablename__ = 'CursosTerminados'
    id = db.Column(db.Integer, primary_key=True)
    microempresario_id = db.Column(db.Integer, db.ForeignKey('microempresarios.id'), nullable=False)
    curso_id = db.Column(db.Integer, db.ForeignKey('cursos.id'), nullable=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
