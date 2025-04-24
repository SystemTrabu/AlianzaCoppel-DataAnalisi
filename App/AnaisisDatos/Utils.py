# Utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import func
from ..models import db , Curso, CursosTerminados, Empresa, MicroEmpresario

class DataProcessor:
    @staticmethod
    def get_microempresarios_data():
        """Obtiene todos los datos de microempresarios con sus cursos terminados"""
        
        # Consulta con agregaciones
        result = db.session.query(
            MicroEmpresario,
            func.count(CursosTerminados.id).label('cursos_completados'),
            func.min(CursosTerminados.fecha).label('primera_actividad'),
            func.max(CursosTerminados.fecha).label('ultima_actividad')
        ).outerjoin(
            CursosTerminados, MicroEmpresario.id == CursosTerminados.microempresario_id
        ).group_by(
            MicroEmpresario.id
        ).all()

        data = []

        for micro, cursos_completados, primera_actividad, ultima_actividad in result:
            tiempo_activacion = None
            tiempo_entre_cursos = None

            # Calcular tiempo desde primera hasta última actividad
            if primera_actividad and ultima_actividad:
                tiempo_activacion = (ultima_actividad - primera_actividad).days

            # Calcular tiempo promedio entre cursos
            if cursos_completados > 1:
                fechas = db.session.query(CursosTerminados.fecha).filter_by(
                    microempresario_id=micro.id
                ).order_by(CursosTerminados.fecha).all()

                fechas = [f[0] for f in fechas]
                diferencias = [(fechas[i] - fechas[i-1]).days for i in range(1, len(fechas))]
                tiempo_entre_cursos = sum(diferencias) / len(diferencias) if diferencias else None

            data.append({
                'id': micro.id,
                'nombre': micro.nombre_empresario,
                'correo': micro.correo,
                'genero': micro.genero,
                'n_telefono': micro.n_telefono,
                'codigo_postal': micro.codigo_postal,
                'webinars': micro.Webinars,
                'colaborador_id': micro.colaborador_id,
                'empresa_id': micro.empresa_id,
                'fecha_registro': micro.fecha_registro,  
                'cursos_completados': cursos_completados,
                'tiempo_activacion': tiempo_activacion,
                'tiempo_entre_cursos': tiempo_entre_cursos,
                'dias_desde_ultima_actividad': (datetime.utcnow() - ultima_actividad).days if ultima_actividad else None
            })


        return pd.DataFrame(data)
    
    @staticmethod
    def get_time_series_data(periodo='diario'):
        """Obtiene datos de series temporales para análisis de forecast
        
        Args:
            periodo: 'diario', 'semanal' o 'mensual'
        """
        # Obtenemos todos los registros de cursos terminados ordenados por fecha
        registros = db.session.query(
            CursosTerminados.fecha
        ).order_by(CursosTerminados.fecha).all()
        
        fechas = [r[0] for r in registros]
        df = pd.DataFrame({'fecha': fechas})
        
        # Agrupamos según el periodo solicitado
        if periodo == 'diario':
            df['fecha'] = df['fecha'].dt.date
        elif periodo == 'semanal':
            df['fecha'] = df['fecha'].dt.to_period('W').dt.to_timestamp()
        elif periodo == 'mensual':
            df['fecha'] = df['fecha'].dt.to_period('M').dt.to_timestamp()
        
        # Contamos los registros por periodo
        time_series = df.groupby('fecha').size().reset_index(name='completados')
        return time_series
    
    @staticmethod
    def get_empresas_data():
        """Obtiene datos agregados por empresa"""
        result = db.session.query(
        Empresa,
        func.count(MicroEmpresario.id).label('total_microempresarios'),
        func.count(CursosTerminados.id).label('total_cursos_completados')
    ).outerjoin(
        MicroEmpresario, Empresa.id == MicroEmpresario.empresa_id  # JOIN explícito
    ).outerjoin(
        CursosTerminados, MicroEmpresario.id == CursosTerminados.microempresario_id  # JOIN explícito
    ).group_by(
        Empresa.id
    ).all()
        
        data = []
        for empresa, total_micro, total_cursos in result:
            data.append({
                'id': empresa.id,
                'nombre': empresa.nombre,
                'tipo': empresa.tipo,
                'fecha_registro': empresa.fecha,
                'total_microempresarios': total_micro,
                'total_cursos_completados': total_cursos,
                'promedio_cursos_por_micro': total_cursos / total_micro if total_micro > 0 else 0
            })
        
        return pd.DataFrame(data)


    
 