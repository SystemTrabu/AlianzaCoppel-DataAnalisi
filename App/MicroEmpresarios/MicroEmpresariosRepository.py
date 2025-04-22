from App.models.MicroEmpresariosModel import MicroEmpresario
from App.models import db


class MicroEmpresarioRepository:

    @staticmethod
    def get_all():
        return MicroEmpresario.query.all()

    @staticmethod
    def get_by_id(id):
        return MicroEmpresario.query.get(id)

    @staticmethod
    def create(data):
        colaborador = MicroEmpresario(**data)
        db.session.add(colaborador)
        db.session.commit()
        return colaborador

    @staticmethod
    def delete(id):
        colaborador = MicroEmpresario.query.get(id)
        if colaborador:
            db.session.delete(colaborador)
            db.session.commit()
        return colaborador
