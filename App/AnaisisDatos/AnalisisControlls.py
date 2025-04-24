from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
import pandas as pd
from .AnalisisService import SegmentacionService, ForecastService, InsightsService, GeneracionDatos
from .Utils import JsonFormatter

analisis_bp = Blueprint('analisis', __name__, url_prefix='/api/analisis')

segmentacion_service = SegmentacionService()
forecast_service = ForecastService()
insights_service = InsightsService()

@analisis_bp.route('/segmentacion', methods=['GET'])
def obtener_segmentacion():
    """Endpoint para obtener la segmentación de microempresarios"""
    try:
        # Obtener parámetro opcional de número de clusters
        n_clusters = request.args.get('n_clusters', default=3, type=int)
        
        # Generar segmentación
        resultados = segmentacion_service.generar_segmentacion(n_clusters=n_clusters)
        
        # Formatear respuesta
        response = JsonFormatter.format_segment_results(resultados)
        
        return jsonify({
            'success': True,
            'data': response
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analisis_bp.route('/forecast', methods=['GET'])
def obtener_forecast():
    """Endpoint para obtener predicciones de crecimiento"""
    try:
        periodo = request.args.get('periodo', default='semanal', type=str)
        periodos_futuros = request.args.get('periodos', default=12, type=int)
        
        if periodo not in ['diario', 'semanal', 'mensual']:
            return jsonify({
                'success': False,
                'error': "El periodo debe ser 'diario', 'semanal' o 'mensual'"
            }), 400
        
        # Generar forecast
        forecast_results, metrics = forecast_service.generar_forecast(
            periodo=periodo, 
            periodos_futuros=periodos_futuros
        )
        
        # Formatear respuesta
        response = JsonFormatter.format_forecast_results(forecast_results, metrics)
        
        return jsonify({
            'success': True,
            'data': response
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analisis_bp.route('/insights', methods=['GET'])
def obtener_insights():
    """Endpoint para obtener insights automáticos"""
    try:
        # Generar insights
        insights = insights_service.generar_insights()
        
        return jsonify({
            'success': True,
            'data': {
                'insights': insights
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analisis_bp.route('/dashboard/resumen', methods=['GET'])
def obtener_resumen_dashboard():
    """Endpoint para obtener un resumen completo para el dashboard"""
    try:
        # Obtener insights
        insights = insights_service.generar_insights()
        print("Paso el generar insights")
        # Obtener segmentación
        segmentacion = segmentacion_service.generar_segmentacion()
        print("Paso generar la segmentacion")
        segmentacion_formateada = JsonFormatter.format_segment_results(segmentacion)
        print("Paso formateo la segmentacion")
        # Obtener forecast
        forecast_results, metrics = forecast_service.generar_forecast(periodo='semanal')
        print("paso forecast")
        forecast_formateado = JsonFormatter.format_forecast_results(forecast_results, metrics)
        print("paso formateo forecast")
        
        return jsonify({
            'success': True,
            'data': {
                'insights': insights,
                'segmentacion': segmentacion_formateada,
                'forecast': forecast_formateado
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Endpoints adicionales para análisis específicos

@analisis_bp.route('/microempresario/<int:id>/segmento', methods=['GET'])
def obtener_segmento_microempresario(id):
    """Endpoint para obtener el segmento de un microempresario específico"""
    try:
        from ..models import db, MicroEmpresario, CursosTerminados
        from sqlalchemy import func
        
        # Obtener datos del microempresario
        micro = MicroEmpresario.query.get(id)
        if not micro:
            return jsonify({
                'success': False,
                'error': f"No se encontró microempresario con ID {id}"
            }), 404
        
        # Obtener conteo de cursos y fechas
        cursos_data = db.session.query(
            func.count(CursosTerminados.id).label('cursos_completados'),
            func.min(CursosTerminados.fecha).label('primera_actividad'),
            func.max(CursosTerminados.fecha).label('ultima_actividad')
        ).filter(CursosTerminados.microempresario_id == id).first()
        
        # Preparar datos para predecir segmento
        micro_data = {
            'id': micro.id,
            'nombre': micro.nombre,
            'codigo_postal': micro.CodigoPostal,
            'webinars': micro.Webinars,
            'colaborador_id': micro.colaborador_id,
            'empresa_id': micro.empresa_id,
            'cursos_completados': cursos_data.cursos_completados if cursos_data else 0,
            'tiempo_activacion': 0,  # Temporalmente 0 si no hay fecha_registro
            'tiempo_entre_cursos': None,
            'dias_desde_ultima_actividad': (datetime.utcnow() - cursos_data.ultima_actividad).days if cursos_data and cursos_data.ultima_actividad else None
        }
        
        # Predecir segmento
        segmento = segmentacion_service.predecir_segmento(micro_data)
        
        return jsonify({
            'success': True,
            'data': {
                'microempresario_id': id,
                'nombre': micro.nombre,
                'segmento': segmento,
                'cursos_completados': micro_data['cursos_completados'],
                'dias_desde_ultima_actividad': micro_data['dias_desde_ultima_actividad']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analisis_bp.route('/colaboradores/efectividad', methods=['GET'])
def obtener_efectividad_colaboradores():
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
    try:
        from .Utils import DataProcessor
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
            total_semana = micro_semana

            
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

            mejor_colaborador = max(colaboradores_data, key=lambda x: x['efectividad'])
            print("Estoy terminando el metodo")
            return jsonify({
                'success': True,
                'data': {
                    'mejor_colaborador': mejor_colaborador,
                    'microempresarios_semana_actual': total_semana.to_dict(orient='records')

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

    return GeneracionDatos.obtenerEmpresarios()


@analisis_bp.route('/getActividad', methods=['GET'])
def getActividad():
    return GeneracionDatos.obtenerActividad()


@analisis_bp.route('/generar/reporte', methods=['GET'])
def GenerarReporte():
    return GeneracionDatos.descargar_reporte() 