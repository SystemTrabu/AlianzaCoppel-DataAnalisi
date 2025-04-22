from .EmpresaRepository import EmpresaRepository
from App.models import db
from App.models.EmpresaModel import Empresa

class EmpresaService:

    @staticmethod
    def listar():
        return  EmpresaRepository.get_all()

    @staticmethod
    def obtener(id):
        return  EmpresaRepository.get_by_id(id)

    @staticmethod
    def crear(data):
        return EmpresaRepository.create(data)
      
      # O puedes devolver ambos si gustas
    @staticmethod
    def eliminar(id):
        return  EmpresaRepository.delete(id)
