from ..models.ColaboradoresModel import Usuario 
from ..models.MicroEmpresariosModel import MicroEmpresario
from ..models.CursosTerminadosModel import CursosTerminados

from flask import jsonify


class ColaboradorService:
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

    def getEmpresarios(id_colaborador):
        microempresarios = MicroEmpresario.query.filter_by(colaborador_id=id_colaborador).all()


        resultado = []
        for m in microempresarios:
            cantidad_cursos = CursosTerminados.query.filter_by(microempresario_id=m.id).count()
            
            resultado.append({
                'id': m.id,
                'nombre': m.nombre,
                'Num_tel': m.N_telefono,
                'CP': m.CodigoPostal,
                'Webinars': m.Webinars,
                'Fecha_Registro': m.empresa.fecha if m.empresa else None,
                'Cursos_terminados': cantidad_cursos
            })

        return resultado
    
    
