from flask import Blueprint, request
from .ColaboradorService import ColaboradorService

colaborador_bp = Blueprint('colaborador', __name__, url_prefix='/api/colaborador')


@colaborador_bp.route('/verificar/<int:id>')
def verificar_colaborador(id):
    #num_empleado = request.args.get('num')
    return ColaboradorService.VerificarColaborador(id)

@colaborador_bp.route('/getColaboradores')
def getColaboradores():
    return ColaboradorService.GetColaboradres()


@colaborador_bp.route('/getEmpresarios/<int:id>')
def getMicroempresarios(id):
    return ColaboradorService.getEmpresarios(id)