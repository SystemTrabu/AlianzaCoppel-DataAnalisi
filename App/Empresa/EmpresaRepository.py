from App.models.EmpresaModel import Empresa
from App.models import db


class EmpresaRepository:

    @staticmethod
    def get_all():
        return Empresa.query.all()

    @staticmethod
    def get_by_id(id):
        return Empresa.query.get(id)

    @staticmethod
    def create(data):
        colaborador = Empresa(**data)
        db.session.add(colaborador)
        db.session.commit()
        return colaborador

    @staticmethod
    def delete(id):
        colaborador = Empresa.query.get(id)
        if colaborador:
            db.session.delete(colaborador)
            db.session.commit()
        return colaborador
