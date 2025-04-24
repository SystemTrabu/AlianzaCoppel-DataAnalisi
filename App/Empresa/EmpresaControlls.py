from flask import Blueprint, request, jsonify
from .EmpresaService import EmpresaService
from .EmpresaShemas import Empresa_schema , Empresas_schema

Empresas_bp = Blueprint('Empresa', __name__, url_prefix='/api/empresa')

@Empresas_bp.route('/', methods=['GET'])
def get_all():
    """
    Obtener todas las empresas 
    ---
    responses:
      200:
        description: Lista de empresas
        schema:
          type: array
          items:
            $ref: '#/definitions/empresas'
    """
    colaboradores =EmpresaService.listar()
    return Empresas_schema.jsonify(colaboradores)

@Empresas_bp.route('/<int:id>', methods=['GET'])
def get_by_id(id):
    """
    Obtener un microempresario por ID
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: empresa encontrado
        schema:
          $ref: '#/definitions/empresas'
      404:
        description: No encontrado
    """
    colaborador = EmpresaService.obtener(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return Empresa_schema.jsonify(colaborador)

@Empresas_bp.route('/', methods=['POST'])
def create():
    """
    Crear nueva empresa
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/empresas'
    responses:
      201:
        description: Creado exitosamente
        schema:
          $ref: '#/definitions/empresas'
    """
    data = request.get_json()
    colaborador = EmpresaService.crear(data)
    return Empresa_schema.jsonify(colaborador), 201

@Empresas_bp.route('/<int:id>', methods=['DELETE'])
def delete(id):
    """
    Eliminar una por ID
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Eliminado exitosamente
      404:
        description: No encontrado
    """
    colaborador = EmpresaService.eliminar(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return jsonify({'mensaje': 'Eliminado'})
