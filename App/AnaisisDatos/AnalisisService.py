# AnalisisService.py
from flask import jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prophet import Prophet
from datetime import datetime, timedelta

from ..models.MicroEmpresariosModel import MicroEmpresario
from ..models.CursosTerminadosModel import CursosTerminados

from .Utils import DataProcessor, JsonFormatter

class SegmentacionService:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.model = None
        self.features = None
        self.data_processor = DataProcessor()
    
    def prepare_features(self, df):
        """Prepara las características para el clustering"""
        # Seleccionamos características relevantes para la segmentación
        features_df = df[['cursos_completados', 'webinars', 'tiempo_activacion', 
                          'tiempo_entre_cursos', 'dias_desde_ultima_actividad']]
        
        # Manejo de valores faltantes
        features_clean = pd.DataFrame(self.imputer.fit_transform(features_df), 
                                    columns=features_df.columns)
        
        # Normalización
        features_scaled = pd.DataFrame(self.scaler.fit_transform(features_clean),
                                     columns=features_clean.columns)
        
        return features_scaled
    
    def generar_segmentacion(self, n_clusters=3):
        """Entrena el modelo de clustering y retorna los resultados"""
        # Obtener datos
        df = DataProcessor.get_microempresarios_data()
        
        # Preparar características
        self.features = self.prepare_features(df)
        
        # Entrenar modelo K-Means
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.model.fit(self.features)
        
        # Añadir etiquetas al dataframe original
        df['cluster'] = self.model.labels_
        
        return self.interpret_clusters(df)
    
    def predecir_segmento(self, microempresario_data):
        """Predice el segmento de un nuevo microempresario"""
        if not self.model:
            self.generar_segmentacion()
        
        # Preparar datos del microempresario
        input_df = pd.DataFrame([microempresario_data])
        input_features = self.prepare_features(input_df)
        
        # Predecir cluster
        cluster = self.model.predict(input_features)[0]
        
        # Mapear a segmento descriptivo
        segment_mapping = {
            0: "Activo", 
            1: "Latente",
            2: "Pasivo"
        }
        
        return segment_mapping.get(cluster, f"Segmento {cluster}")
    
    def interpret_clusters(self, df_with_clusters):
        """Interpreta los clusters y asigna etiquetas descriptivas"""
        # Calcular medias por cluster
        cluster_means = df_with_clusters.groupby('cluster').agg({
            'cursos_completados': 'mean',
            'webinars': 'mean',
            'tiempo_activacion': 'mean',
            'tiempo_entre_cursos': 'mean',
            'dias_desde_ultima_actividad': 'mean'
        })
        
        # Determinar características de cada cluster para asignar nombres
        segment_mapping = {}
        for cluster in cluster_means.index:
            if cluster_means.loc[cluster, 'cursos_completados'] > 2 and \
               cluster_means.loc[cluster, 'dias_desde_ultima_actividad'] < 30:
                segment_name = "Activo"  # Muchos cursos, actividad reciente
            elif cluster_means.loc[cluster, 'cursos_completados'] < 1 and \
                 cluster_means.loc[cluster, 'tiempo_activacion'] > 30:
                segment_name = "Pasivo"  # Pocos cursos, largo tiempo para activarse
            else:
                segment_name = "Latente"  # Entre activo y pasivo
            
            segment_mapping[cluster] = segment_name
        
        # Aplicar mapeo al dataframe
        df_with_clusters['segmento'] = df_with_clusters['cluster'].map(segment_mapping)
        
        # Resumen por segmento
        segment_summary = df_with_clusters.groupby('segmento').agg({
            'id': 'count',
            'cursos_completados': 'mean',
            'webinars': 'mean',
            'tiempo_activacion': 'mean',
            'dias_desde_ultima_actividad': 'mean'
        }).rename(columns={'id': 'cantidad'})
        
        return {
            'df_segmentado': df_with_clusters,
            'segment_summary': segment_summary,
            'segment_mapping': segment_mapping
        }


class ForecastService:
    def __init__(self):
        self.model = None
        self.last_training_date = None
        self.period = 'diario'
    
    def generar_forecast(self, periodo='diario', periodos_futuros=30, seasonality_mode='multiplicative'):
        """Entrena un modelo de Prophet y genera predicciones
        
        Args:
            periodo: 'diario', 'semanal' o 'mensual'
            periodos_futuros: Número de periodos a predecir
            seasonality_mode: 'multiplicative' o 'additive'
            
        Returns:
            Dict con resultados de forecast y métricas
        """
        self.period = periodo
        
        # Obtener datos de series temporales
        time_series = DataProcessor.get_time_series_data(periodo=periodo)
        
        # Preparar dataframe en formato para Prophet (ds, y)
        prophet_df = time_series.rename(columns={'fecha': 'ds', 'completados': 'y'})
        
        # Guardar última fecha de entrenamiento
        self.last_training_date = prophet_df['ds'].max()
        
        # Crear y ajustar modelo
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=periodo=='diario',
            weekly_seasonality=periodo=='diario' or periodo=='semanal',
            yearly_seasonality=True
        )
        
        # Entrenar modelo
        self.model.fit(prophet_df)
        
        # Crear dataframe futuro según el periodo
        if self.period == 'diario':
            future = self.model.make_future_dataframe(periods=periodos_futuros)
        elif self.period == 'semanal':
            future = self.model.make_future_dataframe(periods=periodos_futuros, freq='W')
        elif self.period == 'mensual':
            future = self.model.make_future_dataframe(periods=periodos_futuros, freq='M')
        
        # Realizar predicción
        forecast = self.model.predict(future)
        
        # Filtrar solo predicciones futuras
        future_forecast = forecast[forecast['ds'] > self.last_training_date]
        
        forecast_results = {
            'full_forecast': forecast,
            'future_forecast': future_forecast
        }
        
        # Calcular métricas
        metrics = self.calcular_metricas_crecimiento(forecast_results)
        
        return forecast_results, metrics
    
    def calcular_metricas_crecimiento(self, forecast_data):
        """Calcula métricas de crecimiento basadas en las predicciones"""
        future = forecast_data['future_forecast']
        
        # Calcular total predicho para el próximo periodo
        next_period_total = future['yhat'].sum()
        
        # Calcular tasa de crecimiento respecto al último periodo real
        historical = DataProcessor.get_time_series_data(periodo=self.period)
        last_period_total = historical['completados'].sum()
        
        # Evitar división por cero
        if last_period_total > 0:
            growth_rate = (next_period_total / last_period_total) - 1
        else:
            growth_rate = 0
        
        # Encontrar pico predicho
        peak_date = future.loc[future['yhat'].idxmax(), 'ds']
        peak_value = future['yhat'].max()
        
        return {
            'total_proximo_periodo': next_period_total,
            'tasa_crecimiento': growth_rate,
            'fecha_pico': peak_date,
            'valor_pico': peak_value
        }


class InsightsService:
    def __init__(self):
        self.segmentation_service = SegmentacionService()
        self.forecast_service = ForecastService()
    
    def generar_insights(self):
        """Genera insights automáticos basados en los datos"""
        insights = []
        
        # Obtener datos necesarios
        micro_data = DataProcessor.get_microempresarios_data()
        print("Paso obtener la data del microempresario")
        time_series = DataProcessor.get_time_series_data(periodo='semanal')
        print("Paso el get time_serires_data")
        empresas_data = DataProcessor.get_empresas_data()
        print("pASO get empresas data")
        
        # 1. Insights de segmentación
        segmentation_results = self.segmentation_service.generar_segmentacion()
        print("Paso generar segmentacion")
        segment_df = segmentation_results['df_segmentado']
        segment_summary = segmentation_results['segment_summary']
        
        # Identificar segmento más grande
        largest_segment = segment_summary['cantidad'].idxmax()
        largest_segment_pct = segment_summary.loc[largest_segment, 'cantidad'] / segment_summary['cantidad'].sum()
        
        insights.append({
            'tipo': 'segmentacion',
            'prioridad': 'alta',
            'mensaje': f"El segmento '{largest_segment}' es el más grande, con {largest_segment_pct:.1%} de los microempresarios."
        })
        
        # Insight de segmento con mayor finalización de cursos
        most_active = segment_summary['cursos_completados'].idxmax()
        avg_courses = segment_summary.loc[most_active, 'cursos_completados']
        
        insights.append({
            'tipo': 'segmentacion',
            'prioridad': 'media',
            'mensaje': f"Los microempresarios del segmento '{most_active}' completan un promedio de {avg_courses:.1f} cursos."
        })
        
        # 2. Insights de forecast
        # Generar forecast
        forecast_results, metrics = self.forecast_service.generar_forecast(periodo='semanal')
        
        insights.append({
            'tipo': 'forecast',
            'prioridad': 'alta',
            'mensaje': f"Se prevé un {'crecimiento' if metrics['tasa_crecimiento'] > 0 else 'decrecimiento'} " +
                      f"del {abs(metrics['tasa_crecimiento']):.1%} en cursos completados para las próximas semanas."
        })
        
        # 3. Insights geográficos (por código postal)
        if 'codigo_postal' in micro_data.columns:
            cp_performance = micro_data.groupby('codigo_postal').agg({
                'id': 'count',
                'cursos_completados': ['mean', 'sum']
            })
            
            cp_performance.columns = ['cantidad_usuarios', 'promedio_cursos', 'total_cursos']
            
            if not cp_performance.empty:
                top_cp = cp_performance.sort_values('total_cursos', ascending=False).head(1).index[0]
                
                insights.append({
                    'tipo': 'geografico',
                    'prioridad': 'media',
                    'mensaje': f"El código postal {top_cp} destaca con {cp_performance.loc[top_cp, 'total_cursos']:.0f} cursos completados."
                })
        
        # 4. Insights de colaboradores
        if 'colaborador_id' in micro_data.columns:
            colab_performance = micro_data.groupby('colaborador_id').agg({
                'id': 'count',
                'cursos_completados': ['mean', 'sum']
            })
            
            colab_performance.columns = ['cantidad_usuarios', 'promedio_cursos', 'total_cursos']
            
            if not colab_performance.empty:
                top_colab = colab_performance.sort_values('promedio_cursos', ascending=False).head(1).index[0]
                
                insights.append({
                    'tipo': 'colaborador',
                    'prioridad': 'alta',
                    'mensaje': f"El colaborador ID {top_colab} tiene la mayor efectividad con un promedio de " +
                              f"{colab_performance.loc[top_colab, 'promedio_cursos']:.1f} cursos por microempresario."
                })
        
        # 5. Insights de tendencias recientes
        # Comparar últimas dos semanas
        if len(time_series) >= 2:
            last_week = time_series.iloc[-1]['completados']
            previous_week = time_series.iloc[-2]['completados']
            
            if previous_week > 0:  # Evitar división por cero
                week_change = (last_week / previous_week) - 1
                
                insights.append({
                    'tipo': 'tendencia',
                    'prioridad': 'media',
                    'mensaje': f"La última semana muestra un {'incremento' if week_change > 0 else 'decremento'} " +
                              f"del {abs(week_change):.1%} en cursos completados."
                })
        
        # 6. Insight de empresas
        if not empresas_data.empty:
            # Empresa con mayor promedio de cursos por microempresario
            top_empresa = empresas_data.sort_values('promedio_cursos_por_micro', ascending=False).head(1)
            
            if not top_empresa.empty:
                emp_id = top_empresa.index[0]
                emp_name = top_empresa.loc[emp_id, 'nombre']
                
                insights.append({
                    'tipo': 'empresa',
                    'prioridad': 'media',
                    'mensaje': f"La empresa '{emp_name}' tiene el mejor rendimiento con {top_empresa.loc[emp_id, 'promedio_cursos_por_micro']:.1f} " +
                              f"cursos por microempresario."
                })
        
        # Ordenar insights por prioridad
        priority_map = {'alta': 0, 'media': 1, 'baja': 2}
        insights.sort(key=lambda x: priority_map[x['prioridad']])
        
        return insights
    

class GeneracionDatos:
    def obtenerEmpresarios():
        data_return=[]
        microempresarios_total=MicroEmpresario.query.count()
        

        # Fecha actual
        hoy = datetime.now()

        # Primer día del mes actual
        inicio_mes = datetime(hoy.year, hoy.month, 1)

        if hoy.month == 12:
            siguiente_mes = datetime(hoy.year + 1, 1, 1)
        else:
            siguiente_mes = datetime(hoy.year, hoy.month + 1, 1)

        fin_mes = siguiente_mes - timedelta(seconds=1)

        # Filtrar los microempresarios registrados en este mes
        microempresarios_total_mes = MicroEmpresario.query.filter(
            MicroEmpresario.fecha_registro >= inicio_mes,
            MicroEmpresario.fecha_registro <= fin_mes
        ).count()


        

        df_cursos = pd.DataFrame([c.__dict__ for c in CursosTerminados.query.all()])
        df_empresarios = pd.DataFrame([m.__dict__ for m in MicroEmpresario.query.all()])

        df_cursos.drop(columns=['_sa_instance_state'], inplace=True, errors='ignore')
        df_empresarios.drop(columns=['_sa_instance_state'], inplace=True, errors='ignore')


        cursos_count = df_cursos.groupby('microempresario_id').size().reset_index(name='cursos_completados')
        
        df = pd.merge(df_empresarios, cursos_count, left_on='id', right_on='microempresario_id', how='left')
        df['cursos_completados'] = df['cursos_completados'].fillna(0)

        conditions = [
            (df['cursos_completados'] >= 20),
            (df['cursos_completados'] >= 6) & (df['cursos_completados'] <= 19),
            (df['cursos_completados'] <= 5)
        ]
        choices = ['activo', 'latente', 'inactivo']
        df['actividad'] = np.select(conditions, choices, default='inactivo')

        actividad_counts = df['actividad'].value_counts().to_dict()

        actividad_total = {
            'activo': actividad_counts.get('activo', 0),
            'latente': actividad_counts.get('latente', 0),
            'inactivo': actividad_counts.get('inactivo', 0)
        }

        data_return.append({
            'total_micro':microempresarios_total,
            'total_micro_mes': microempresarios_total_mes,
            'distribucion_actividad': actividad_total
        })


        return jsonify(data_return)