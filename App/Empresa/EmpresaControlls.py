from flask import Blueprint, request, jsonify
from .EmpresaService import EmpresaService
from .EmpresaShemas import Empresa_schema , Empresas_schema

Empresas_bp = Blueprint('Empresa', __name__, url_prefix='/api/empresa')

@Empresas_bp.route('/', methods=['GET'])
def get_all():
    colaboradores =EmpresaService.listar()
    return Empresas_schema.jsonify(colaboradores)

@Empresas_bp.route('/<int:id>', methods=['GET'])
def get_by_id(id):
    colaborador = EmpresaService.obtener(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return Empresa_schema.jsonify(colaborador)

@Empresas_bp.route('/', methods=['POST'])
def create():
    data = request.get_json()
    colaborador = EmpresaService.crear(data)
    return Empresa_schema.jsonify(colaborador), 201

@Empresas_bp.route('/<int:id>', methods=['DELETE'])
def delete(id):
    colaborador = EmpresaService.eliminar(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return jsonify({'mensaje': 'Eliminado'})
