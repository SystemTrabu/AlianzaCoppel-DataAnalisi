from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
import pandas as pd

from App.models import ColaboradoresModel
from .AnalisisService import SegmentacionService, ForecastService, InsightsService, GeneracionDatos


analisis_bp = Blueprint('analisis', __name__, url_prefix='/api/analisis')


@analisis_bp.route('/colaboradores/efectividad/<int:cantidad>', methods=['GET'])
def obtener_efectividad_colaboradores_cantidad(cantidad):
    try:
        from .Utils import DataProcessor
        micro_data = DataProcessor.get_microempresarios_data()
        if 'colaborador_id' in micro_data.columns:
            colab_performance = micro_data.groupby('colaborador_id').agg({
                'id': 'count',
                'cursos_completados': ['mean', 'sum']
            })
            
            colab_performance.columns = ['cantidad_usuarios', 'promedio_cursos', 'total_cursos']
            
            colaboradores_data = []
            contador=0
            for colaborador_id in colab_performance.index:
                colaboradores_data.append({
                    'colaborador_id': int(colaborador_id),
                    'microempresarios_asignados': int(colab_performance.loc[colaborador_id, 'cantidad_usuarios']),
                    'cursos_promedio': float(colab_performance.loc[colaborador_id, 'promedio_cursos']),
                    'total_cursos': int(colab_performance.loc[colaborador_id, 'total_cursos']),
                    'efectividad': float(colab_performance.loc[colaborador_id, 'promedio_cursos'])
                })
                contador+=1
                if contador== cantidad:
                    break
            
            colaboradores_data = sorted(colaboradores_data, key=lambda x: x['efectividad'], reverse=True)
            
            return jsonify({
                'success': True,
                'data': {
                    'colaboradores': colaboradores_data
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': "No se encontraron datos de colaboradores"
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analisis_bp.route('/colaboradores/efectividad', methods=['GET'])
def obtener_efectividad_colaboradores():
    """
    Genera un ranking de los colaboradores
    ---
    tags:
      - Analisis
    responses:
      200:
        description: Ranking de colaboradores por efectividad
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                colaboradores:
                  type: array
                  items:
                    type: object
                    properties:
                      colaborador_id:
                        type: integer
                      microempresarios_asignados:
                        type: integer
                      cursos_promedio:
                        type: number
                      total_cursos:
                        type: integer
                      efectividad:
                        type: number
    """
    try:
        from .Utils import DataProcessor
        micro_data = DataProcessor.get_microempresarios_data()
        if 'colaborador_id' in micro_data.columns:
            colab_performance = micro_data.groupby('colaborador_id').agg({
                'id': 'count',
                'cursos_completados': ['mean', 'sum']
            })
            
            colab_performance.columns = ['cantidad_usuarios', 'promedio_cursos', 'total_cursos']
            
            colaboradores_data = []
            for colaborador_id in colab_performance.index:
                colaboradores_data.append({
                    'colaborador_id': int(colaborador_id),
                    'microempresarios_asignados': int(colab_performance.loc[colaborador_id, 'cantidad_usuarios']),
                    'cursos_promedio': float(colab_performance.loc[colaborador_id, 'promedio_cursos']),
                    'total_cursos': int(colab_performance.loc[colaborador_id, 'total_cursos']),
                    'efectividad': float(colab_performance.loc[colaborador_id, 'promedio_cursos'])
                })
            
            colaboradores_data = sorted(colaboradores_data, key=lambda x: x['efectividad'], reverse=True)
            
            return jsonify({
                'success': True,
                'data': {
                    'colaboradores': colaboradores_data
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': "No se encontraron datos de colaboradores"
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

@analisis_bp.route("/mejorefectividad")
def ObtenerMejorEfectividad():
    """
    Mejor colaborador top 1
    ---
    tags:
      - Analisis
    responses:
      200:
        description: Mejor colaborador top 1 y sus microempresarios con datos de esa semana
        schema:
          type: object
          properties:
            success:
              type: boolean
            data:
              type: object
              properties:
                mejor_colaborador:
                  type: object
                  properties:
                    colaborador_id:
                      type: integer
                    microempresarios_asignados:
                      type: integer
                    cursos_promedio:
                      type: number
                    total_cursos:
                      type: integer
                    efectividad:
                      type: number
                microempresarios_semana_actual:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      nombre:
                        type: string
                      correo:
                        type: string
                      genero:
                        type: string
                      n_telefono:
                        type: string
                      codigo_postal:
                        type: integer
                      colaborador_id:
                        type: integer
                      empresa_id:
                        type: integer
                      fecha_registro:
                        type: string
                        format: date-time
    """
    try:
        from .Utils import DataProcessor
        import pandas as pd
        from datetime import datetime, timedelta
        from flask import jsonify

        micro_data = DataProcessor.get_microempresarios_data()
        
        if 'colaborador_id' in micro_data.columns:
            micro_data['fecha_registro'] = pd.to_datetime(micro_data['fecha_registro'])

            hoy = datetime.now()
            inicio_semana = hoy - timedelta(days=hoy.weekday())  
            fin_semana = inicio_semana + timedelta(days=6)      
            micro_semana = micro_data[
                (micro_data['fecha_registro'] >= inicio_semana) &
                (micro_data['fecha_registro'] <= fin_semana)
            ]
            micro_semanal_por_colaborador = micro_semana.groupby('colaborador_id')['id'].count()

            colab_performance = micro_data.groupby('colaborador_id').agg({
                'id': 'count',
                'cursos_completados': ['mean', 'sum']
            })
            colab_performance.columns = ['cantidad_usuarios', 'promedio_cursos', 'total_cursos']
            
            colaboradores_data = []
            for colaborador_id in colab_performance.index:
                colaborador = ColaboradoresModel.Usuario.query.filter_by(id=colaborador_id).first()
                colaboradores_data.append({
                    'colaborador_id': int(colaborador_id),
                    'nombre_colaborador': colaborador.nombre,
                    'microempresarios_semanal': int(micro_semanal_por_colaborador.get(colaborador_id, 0)),
                    'cursos_promedio': float(colab_performance.loc[colaborador_id, 'promedio_cursos']),
                    'total_cursos': int(colab_performance.loc[colaborador_id, 'total_cursos']),
                    'efectividad': float(colab_performance.loc[colaborador_id, 'promedio_cursos'])
                })

            mejor_colaborador = max(colaboradores_data, key=lambda x: x['efectividad'])

            return jsonify({
                'success': True,
                'data': {
                    'mejor_colaborador': mejor_colaborador
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': "No se encontraron datos de colaboradores"
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@analisis_bp.route('/getEmpresarios', methods=["GET"])
def getEmpresarios():
    """
    Obtener todos los empresarios que son activos, inactivos y latentes
    ---
    tags:
      - Analisis
    responses:
      200:
        description: Distribución de actividad de empresarios
        schema:
          type: object
          properties:
            activos:
              type: integer
            inactivos:
              type: integer
            latentes:
              type: integer
            total:
              type: integer
    """
    return GeneracionDatos.obtenerEmpresarios()


# @analisis_bp.route('/getActividad', methods=['GET'])
# def getActividad():
#     return GeneracionDatos.obtenerActividad()


@analisis_bp.route('/generar/reporte', methods=['GET'])
def GenerarReporte():
    """
    Genera un reporte con diagramas y datos
    ---
    responses:
      200:
        description: Genera y descarga un archivo PDF con diagramas y datos
        content:
          application/pdf:
            schema:
              type: string
              format: binary
      500:
        description: Error al generar el reporte
    """
    return GeneracionDatos.descargar_reporte() 