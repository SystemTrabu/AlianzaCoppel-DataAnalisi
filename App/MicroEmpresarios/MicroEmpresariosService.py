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
    def eliminar(id):
        return  MicroEmpresarioRepository.delete(id)

    @staticmethod
    def crear(data):
        # 1. Crear primero el negocio (empresa)
        negocio = Empresa(
            nombreempresa=data['nombreempresa'],
            tipo=data['tipo'],
            fecha=data['fecha']
        )
        db.session.add(negocio)
        db.session.flush()  # Genera el ID de la empresa sin hacer commit

        # 2. Crear MicroEmpresario con empresa_id generado
        empresario = MicroEmpresario(
            nombre=data['nombre'],
            CodigoPostal=data['CodigoPostal'],
            N_telefono=data['N_telefono'],
            Webinars=data['Webinars'],
            colaborador_id=data.get('colaborador_id'),  # Usa .get por si no lo mandan
            empresa_id=negocio.id
        )

        db.session.add(empresario)
        db.session.commit()

        return empresario
