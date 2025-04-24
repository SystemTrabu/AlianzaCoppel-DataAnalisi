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

    def crear(data):
        # 1. Crear primero la empresa (negocio)
        negocio = Empresa(
            nombre_empresa=data['nombre_empresa'],
            tipo_empresa=data['tipo_empresa'],
            n_empleados=data.get('n_empleados'),
            ingresos_semanales=data.get('ingresos_semanales'),
            nivel_madurez=data.get('nivel_madurez'),
            negocio_familiar=data.get('negocio_familiar'),
            antiguedad=data.get('antiguedad')
        )
        db.session.add(negocio)
        db.session.flush()  # Para obtener el ID de empresa

        # 2. Crear MicroEmpresario con empresa_id generado
        empresario = MicroEmpresario(
            nombre_empresario=data['nombre_empresario'],
            genero=data.get('genero'),
            correo=data.get('correo'),
            fecha_nacimiento=data.get('fecha_nacimiento'),
            fecha_registro=data.get('fecha_registro'),
            n_telefono=data['n_telefono'],
            codigo_postal=data['codigo_postal'],
            Webinars=data['Webinars'],
            nivel_educativo=data.get('nivel_educativo'),
            colaborador_id=data.get('colaborador_id'),
            estado=data.get('estado'),
            empresa_id=negocio.id  # ID generado en flush
        )
        db.session.add(empresario)
        db.session.commit()

