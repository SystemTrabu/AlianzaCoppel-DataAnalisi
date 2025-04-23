from flask import Blueprint, request, jsonify
from .MicroEmpresariosService import MicroEmpresariosService
from .MicroEmpresariosSchemas import MicroEmpresario_schema, MicroEmpresarios_schema

MicroEmpresario_bp = Blueprint('Empresarios', __name__, url_prefix='/api/empresarios')

@MicroEmpresario_bp.route('/', methods=['GET'])
def get_all():
    colaboradores =MicroEmpresariosService.listar()
    return MicroEmpresarios_schema.jsonify(colaboradores)

@MicroEmpresario_bp.route('/<int:id>', methods=['GET'])
def get_by_id(id):
    colaborador = MicroEmpresariosService.obtener(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return MicroEmpresario_schema.jsonify(colaborador)

@MicroEmpresario_bp.route('/agregar', methods=['POST'])
def create():
    data = request.get_json()
    colaborador = MicroEmpresariosService.crear(data)
    return MicroEmpresario_schema.jsonify(colaborador), 201

@MicroEmpresario_bp.route('/<int:id>', methods=['DELETE'])
def delete(id):
    colaborador = MicroEmpresariosService.eliminar(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return jsonify({'mensaje': 'Eliminado'})
