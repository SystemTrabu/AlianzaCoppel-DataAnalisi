from flask import Blueprint, request, jsonify
from .MicroEmpresariosService import MicroEmpresariosService
from .MicroEmpresariosSchemas import MicroEmpresario_schema, MicroEmpresarios_schema

MicroEmpresario_bp = Blueprint('Empresarios', __name__)

@MicroEmpresario_bp.route('/Empresarios', methods=['GET'])
def get_all():
    colaboradores =MicroEmpresariosService.listar()
    return MicroEmpresarios_schema.jsonify(colaboradores)

@MicroEmpresario_bp.route('/Empresarios/<int:id>', methods=['GET'])
def get_by_id(id):
    colaborador = MicroEmpresariosService.obtener(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return MicroEmpresario_schema.jsonify(colaborador)

@MicroEmpresario_bp.route('/Empresarios', methods=['POST'])
def create():
    data = request.get_json()
    colaborador = MicroEmpresariosService.crear(data)
    return MicroEmpresario_schema.jsonify(colaborador), 201

@MicroEmpresario_bp.route('/Empresarios/<int:id>', methods=['DELETE'])
def delete(id):
    colaborador = MicroEmpresariosService.eliminar(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return jsonify({'mensaje': 'Eliminado'})
