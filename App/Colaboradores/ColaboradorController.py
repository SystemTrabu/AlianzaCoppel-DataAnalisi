from flask import Blueprint, request
from .ColaboradorService import VerificarColaborador

colaborador_bp = Blueprint('colaborador', __name__, url_prefix='/api/colaborador')


@colaborador_bp.route('/verificar')
def verificar_colaborador():
    num_empleado = request.args.get('num')
    print(f"El numero {num_empleado}")
    return VerificarColaborador(num_empleado)