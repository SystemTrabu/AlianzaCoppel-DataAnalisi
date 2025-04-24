# AnalisisService.py
import io
import os
from flask import jsonify, make_response
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prophet import Prophet
from datetime import datetime, timedelta

from ..models.MicroEmpresariosModel import MicroEmpresario
from ..models.CursosTerminadosModel import CursosTerminados
from ..models.EmpresaModel import Empresa

import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
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
    

    def obtenerActividad():
        micro_activos=MicroEmpresario.query.filter_by(actividad="activo")
    

    def descargar_reporte():
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        elementos = []
        
        estilos = getSampleStyleSheet()
        
        titulo = Paragraph("Reporte AlianzaCoppel", estilos["Title"])
        elementos.append(titulo)
        
        texto = Paragraph("Reporte detallado de microempresarios según su nivel de actividad", estilos["Normal"])
        elementos.append(texto)
        
        elementos.append(Spacer(1, 20))

        df_cursos = pd.DataFrame([c.__dict__ for c in CursosTerminados.query.all()])
        df_empresarios = pd.DataFrame([m.__dict__ for m in MicroEmpresario.query.all()])
        df_empresas=pd.DataFrame([x.__dict__ for x in Empresa.query.all()])

        df_cursos.drop(columns=['_sa_instance_state'], inplace=True, errors='ignore')
        df_empresarios.drop(columns=['_sa_instance_state'], inplace=True, errors='ignore')
        df_empresas.drop(columns=['_sa_instance_state'], inplace=True, errors='ignore')

        cursos_count = df_cursos.groupby('microempresario_id').size().reset_index(name='cursos_completados')
        
        df = pd.merge(df_empresarios, cursos_count, left_on='id', right_on='microempresario_id', how='left')
        df = pd.merge(df, df_empresas, left_on='empresa_id', right_on='id', how='left')

        df['cursos_completados'] = df['cursos_completados'].fillna(0)

        df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'])
        df['edad'] = (pd.to_datetime('today') - df['fecha_nacimiento']).dt.days // 365
        conditions = [
            (df['cursos_completados'] >= 20),
            (df['cursos_completados'] >= 6) & (df['cursos_completados'] <= 19),
            (df['cursos_completados'] <= 5)
        ]
        choices = ['activo', 'latente', 'inactivo']
        df['actividad'] = np.select(conditions, choices, default='inactivo')

        temp_files = []

        subtitulo = Paragraph("1. Distribución de Microempresarios por Nivel de Actividad", estilos["Heading2"])
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 10))
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='actividad', data=df, order=['activo', 'latente', 'inactivo'])
        plt.title('Distribución de Microempresarios por Nivel de Actividad')
        plt.xlabel('Nivel de Actividad')
        plt.ylabel('Cantidad de Microempresarios')
        
        # Agregar valores encima de cada barra
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom')
        
        # Guardar la gráfica en un archivo temporal
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            temp_files.append(temp_filename)
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Agregar la imagen al PDF
        imagen = Image(temp_filename, width=6*inch, height=3.5*inch)
        elementos.append(imagen)
        
        # Agregar un resumen de texto después de la gráfica
        elementos.append(Spacer(1, 15))
        
        # Contar microempresarios por categoría
        activos = df[df['actividad'] == 'activo'].shape[0]
        latentes = df[df['actividad'] == 'latente'].shape[0]
        inactivos = df[df['actividad'] == 'inactivo'].shape[0]
        total = df.shape[0]
        
        resumen = Paragraph(f"""
        <b>Resumen de Actividad:</b><br/>
        Total de microempresarios: {total}<br/>
        - Microempresarios activos (≥20 cursos): {activos} ({activos/total*100:.1f}%)<br/>
        - Microempresarios latentes (6-19 cursos): {latentes} ({latentes/total*100:.1f}%)<br/>
        - Microempresarios inactivos (≤5 cursos): {inactivos} ({inactivos/total*100:.1f}%)
        """, estilos["Normal"])
        
        elementos.append(resumen)
        #elementos.append(PageBreak())
        
        # 2. Distribución de Edad por Nivel de Actividad
        subtitulo = Paragraph("2. Distribución de Edad por Nivel de Actividad", estilos["Heading2"])
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 10))
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='actividad', y='edad', data=df, order=['activo', 'latente', 'inactivo'])
        plt.title('Distribución de Edad por Nivel de Actividad')
        plt.xlabel('Nivel de Actividad')
        plt.ylabel('Edad')
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            temp_files.append(temp_filename)
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        imagen = Image(temp_filename, width=6.5*inch, height=3.5*inch)
        elementos.append(imagen)
        elementos.append(Spacer(1, 15))
        
        # 3. Ingresos semanales por actividad
        subtitulo = Paragraph("3. Distribución de Ingresos Semanales por Nivel de Actividad", estilos["Heading2"])
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 10))
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='actividad', y='ingresos_semanales', data=df, order=['activo', 'latente', 'inactivo'])
        plt.title('Distribución de Ingresos Semanales por Nivel de Actividad')
        plt.xlabel('Nivel de Actividad')
        plt.ylabel('Ingresos Semanales')
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            temp_files.append(temp_filename)
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        imagen = Image(temp_filename, width=6.5*inch, height=3.5*inch)
        elementos.append(imagen)
        #elementos.append(PageBreak())
        
        # 4. Participación en webinars
        subtitulo = Paragraph("4. Participación en Webinars por Nivel de Actividad", estilos["Heading2"])
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 10))
        
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Webinars', hue='actividad', data=df, 
                    hue_order=['activo', 'latente', 'inactivo'],
                    palette="Set2")
        plt.title('Participación en Webinars por Nivel de Actividad')
        plt.xlabel('Número de Webinars Asistidos')
        plt.ylabel('Cantidad de Microempresarios')
        plt.legend(title='Nivel de Actividad')
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            temp_files.append(temp_filename)
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        imagen = Image(temp_filename, width=6.5*inch, height=3.5*inch)
        elementos.append(imagen)
        elementos.append(Spacer(1, 15))
        
        # 5. Tipo de empresa por actividad
        subtitulo = Paragraph("5. Distribución de Tipos de Empresa por Nivel de Actividad", estilos["Heading2"])
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 10))
        
        plt.figure(figsize=(14, 7))
        df_temp = df.groupby(['tipo_empresa', 'actividad']).size().unstack()
        df_temp.plot(kind='bar', stacked=True)
        plt.title('Distribución de Tipos de Empresa por Nivel de Actividad')
        plt.xlabel('Tipo de Empresa')
        plt.ylabel('Cantidad de Microempresarios')
        plt.legend(title='Nivel de Actividad')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            temp_files.append(temp_filename)
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        imagen = Image(temp_filename, width=6.5*inch, height=3.5*inch)
        elementos.append(imagen)
        elementos.append(PageBreak())
        
        # 6. Nivel educativo por actividad
        subtitulo = Paragraph("6. Distribución por Nivel Educativo", estilos["Heading2"])
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 10))
        
        plt.figure(figsize=(12, 6))
        sns.countplot(x='nivel_educativo', hue='actividad', data=df,
                    hue_order=['activo', 'latente', 'inactivo'],
                    order=df['nivel_educativo'].value_counts().index)
        plt.title('Distribución por Nivel Educativo')
        plt.xlabel('Nivel Educativo')
        plt.ylabel('Cantidad')
        plt.legend(title='Nivel de Actividad')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            temp_files.append(temp_filename)
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        imagen = Image(temp_filename, width=6.5*inch, height=3.5*inch)
        elementos.append(imagen)
        elementos.append(Spacer(1, 15))
        
        # 7. Comparación de géneros
        subtitulo = Paragraph("7. Distribución por Género y Nivel de Actividad", estilos["Heading2"])
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 10))
        
        plt.figure(figsize=(12, 6))
        sns.countplot(x='genero', hue='actividad', data=df,
                    hue_order=['activo', 'latente', 'inactivo'],
                    order=['hombre', 'mujer', 'otro'])
        plt.title('Distribución por Género y Nivel de Actividad')
        plt.xlabel('Género')
        plt.ylabel('Cantidad de Microempresarios')
        plt.legend(title='Nivel de Actividad')
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_filename = tmp.name
            temp_files.append(temp_filename)
        
        plt.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        imagen = Image(temp_filename, width=6.5*inch, height=3.5*inch)
        elementos.append(imagen)
        
        doc.build(elementos)
        
        # Eliminar los archivos temporales
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        # Preparar el buffer para lectura
        buffer.seek(0)
        
        # Crear una respuesta con el PDF
        response = make_response(buffer.getvalue())
        
        # Establecer las cabeceras para la descarga
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=reporte_alianzacoppel.pdf'
        
        return response