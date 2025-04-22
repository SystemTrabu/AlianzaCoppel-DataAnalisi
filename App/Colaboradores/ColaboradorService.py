from ..models.ColaboradoresModel import Usuario 
from flask import jsonify

def VerificarColaborador(num_empleado):
    colaborador = Usuario.query.filter_by(N_Empleado=num_empleado).first()
    
    if colaborador:
        return jsonify({
            'existe': True,
            'id': colaborador.id,
            'nombre': colaborador.nombre
        })
    else:
        return jsonify({'existe': False}), 404



def GetColaboradres():
    usuarios = Usuario.query.all()
    
    usuarios_dict = [{
        'id': u.id,
        'Num_Empleado': u.N_Empleado,
        'nombre': u.nombre
    } for u in usuarios]

    return jsonify(usuarios_dict)
