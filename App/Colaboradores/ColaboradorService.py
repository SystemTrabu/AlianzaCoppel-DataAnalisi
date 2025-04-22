from ..models.ColaboradoresModel import Usuario 
from flask import jsonify

def VerificarColaborador(num_empleado):
    colaborador = Usuario.query.filter_by(N_Empleado=num_empleado).first()
    
    if colaborador:
        return jsonify({
            'existe': True,
        })
    else:
        return jsonify({'existe': False}), 404
