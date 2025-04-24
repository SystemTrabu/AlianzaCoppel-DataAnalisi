from flask import Blueprint, request, jsonify
from .MicroEmpresariosService import MicroEmpresariosService
from .MicroEmpresariosSchemas import MicroEmpresario_schema, MicroEmpresarios_schema
# from flasgger import swag_from  # <-- Solo si luego decides usar decoradores con archivos .yml

MicroEmpresario_bp = Blueprint('Empresarios', __name__, url_prefix='/api/empresarios')

@MicroEmpresario_bp.route('/', methods=['GET'])
def get_all():
    """
    Obtener todos los microempresarios
    ---
    responses:
      200:
        description: Lista de microempresarios
        schema:
          type: array
          items:
            $ref: '#/definitions/MicroEmpresarioGET'
    """
    colaboradores = MicroEmpresariosService.listar()
    return MicroEmpresarios_schema.jsonify(colaboradores)


@MicroEmpresario_bp.route('/<int:id>', methods=['GET'])
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
        description: Microempresario encontrado
        schema:
          $ref: '#/definitions/MicroEmpresarioGET'
      404:
        description: No encontrado
    """
    colaborador = MicroEmpresariosService.obtener(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return MicroEmpresario_schema.jsonify(colaborador)


@MicroEmpresario_bp.route('/', methods=['POST'])
def create():
    """
    Crear un nuevo microempresario
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/RegistroCompletoPOST'
    responses:
      201:
        description: Creado exitosamente
        schema:
          $ref: '#/definitions/RegistroCompletoPOST'
    """
    data = request.get_json()
    colaborador = MicroEmpresariosService.crear(data)
    return MicroEmpresario_schema.jsonify(colaborador), 201


@MicroEmpresario_bp.route('/<int:id>', methods=['DELETE'])
def delete(id):
    """
    Eliminar un microempresario por ID
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
    colaborador = MicroEmpresariosService.eliminar(id)
    if not colaborador:
        return jsonify({'mensaje': 'No encontrado'}), 404
    return jsonify({'mensaje': 'Eliminado'})
