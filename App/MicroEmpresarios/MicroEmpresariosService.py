from .MicroEmpresariosRepository import MicroEmpresarioRepository
from App.models import db
from App.models.MicroEmpresariosModel import MicroEmpresario
from App.models.EmpresaModel import Empresa

class MicroEmpresariosService:

    @staticmethod
    def listar():
        return  MicroEmpresarioRepository.get_all()

    @staticmethod
    def obtener(id):
        return  MicroEmpresarioRepository.get_by_id(id)

    @staticmethod
    def crear(data):
        # 1. Extraer datos para MicroEmpresario
        empresario = MicroEmpresario(
            nombre=data['nombre'],
            CodigoPostal=data['CodigoPostal'],
            N_telefono=data['N_telefono'],
            Webinars=data['Webinars'],
            
        )

        # 2. Crear primero el empresario
        db.session.add(empresario)
        db.session.flush()  # Esto genera el ID sin hacer commit

        # 3. Crear el negocio y asignar el ID del empresario
        negocio = Empresa(
            nombre=data['negocio'],
            direccion=data['direccion_negocio'],
            ingresos=data['ingresos'],
            empresario_id=empresario.id
        )

        db.session.add(negocio)
        db.session.commit()

        return empresario  # O puedes devolver ambos si gustas
    @staticmethod
    def eliminar(id):
        return  MicroEmpresarioRepository.delete(id)
