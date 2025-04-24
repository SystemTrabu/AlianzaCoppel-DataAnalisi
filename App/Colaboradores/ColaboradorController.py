from flask import Blueprint, request
from .ColaboradorService import ColaboradorService

colaborador_bp = Blueprint('colaborador', __name__, url_prefix='/api/colaborador')


@colaborador_bp.route('/verificar/<int:id>')
def verificar_colaborador(id):
    """
    verificar colaborador atravez del n de control
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: colaborador encontrado
        schema:
          $ref: '#/definitions/colaboradoresGet'
      404:
        description: No encontrado
    """
    #num_empleado = request.args.get('num')
    return ColaboradorService.VerificarColaborador(id)

@colaborador_bp.route('/getColaboradores')
def getColaboradores():
    """
    Obtener todos los colaboradores 
    ---
    responses:
      200:
        description: Lista de  colaboradores
        schema:
          type: array
          items:
            $ref: '#/definitions/colaboradores'
    """
    return ColaboradorService.GetColaboradres()


@colaborador_bp.route('/getEmpresarios/<int:id>')
def getMicroempresarios(id):
    """
    Obtener todos los empresarios  con la id del colaborador
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: empresarios encontrados
        schema:
          $ref: '#/definitions/colaboradoresGetEmpresarios'
      404:
        description: No encontrado
    """
    return ColaboradorService.getEmpresarios(id)