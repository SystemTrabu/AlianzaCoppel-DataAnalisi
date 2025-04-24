import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import joblib
import json

# Configuración de visualización
plt.style.use('ggplot')
sns.set_palette("husl")

# 1. Conexión a la base de datos y extracción de datos
engine = create_engine('postgresql://postgres:HRxlTXYjtnYgTVbehhRqwnCEnopFFegE@shuttle.proxy.rlwy.net:40252/railway')

# Cargar datos
df_empresarios = pd.read_sql("SELECT * FROM microempresario", engine)
df_empresas = pd.read_sql("SELECT * FROM empresas", engine)
df_cursos = pd.read_sql('SELECT * FROM "CursosTerminados"', engine)

# Preparación de los datos
cursos_count = df_cursos.groupby('microempresario_id').size().reset_index(name='cursos_completados')

df = pd.merge(df_empresarios, cursos_count, left_on='id', right_on='microempresario_id', how='left')
df = pd.merge(df, df_empresas, left_on='empresa_id', right_on='id', how='left', suffixes=('', '_empresa'))

df['cursos_completados'] = df['cursos_completados'].fillna(0)

# Calcular edad
df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'])
df['edad'] = (pd.to_datetime('today') - df['fecha_nacimiento']).dt.days // 365

# Clasificar actividad
conditions = [
    (df['cursos_completados'] >= 20),
    (df['cursos_completados'] >= 6) & (df['cursos_completados'] <= 19),
    (df['cursos_completados'] <= 5)
]
choices = ['activo', 'latente', 'inactivo']
df['actividad'] = np.select(conditions, choices, default='inactivo')

# =============================================
# VISUALIZACIONES ANTES DEL MODELADO
# =============================================

# 1. Distribución de actividades
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='actividad', data=df, order=['activo', 'latente', 'inactivo'])
plt.title('Distribución de Microempresarios por Nivel de Actividad')
plt.xlabel('Nivel de Actividad')
plt.ylabel('Cantidad de Microempresarios')

# Agregar etiquetas con los valores
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

plt.tight_layout()
plt.show()

# 2. Distribución por edad y actividad
plt.figure(figsize=(12, 6))
sns.boxplot(x='actividad', y='edad', data=df, order=['activo', 'latente', 'inactivo'])
plt.title('Distribución de Edad por Nivel de Actividad')
plt.xlabel('Nivel de Actividad')
plt.ylabel('Edad')
plt.tight_layout()
plt.show()

# 3. Distribución de ingresos por actividad
plt.figure(figsize=(12, 6))
sns.boxplot(x='actividad', y='ingresos_semanales', data=df, order=['activo', 'latente', 'inactivo'])
plt.title('Distribución de Ingresos Semanales por Nivel de Actividad')
plt.xlabel('Nivel de Actividad')
plt.ylabel('Ingresos Semanales')
plt.tight_layout()
plt.show()

# =============================================
# MODELADO PREDICTIVO
# =============================================

# 1. Definir las características relevantes
features = df[[
    'edad', 'nivel_educativo', 
    'tipo_empresa', 'nivel_madurez', 'n_empleados', 
    'negocio_familiar', 'ingresos_semanales', 'antiguedad'
]]

target = df['actividad']

# 2. Definir transformadores para preprocesamiento
categorical_cols = ['nivel_educativo', 'tipo_empresa', 'nivel_madurez', 'negocio_familiar']
numeric_cols = ['edad', 'n_empleados', 'ingresos_semanales', 'antiguedad']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 3. Configurar el pipeline completo de modelado
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target)

# 5. Entrenar el modelo
print("Entrenando el modelo...")
model_pipeline.fit(X_train, y_train)

# 6. Evaluar el rendimiento del modelo
print("\n--- Evaluación del Modelo ---")
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['activo', 'inactivo', 'latente'],
            yticklabels=['activo', 'inactivo', 'latente'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# 7. Guardar el modelo entrenado
joblib.dump(model_pipeline, 'modelo_actividad_microempresarios.pkl')
print("\nModelo guardado como 'modelo_actividad_microempresarios.pkl'")

# =============================================
# FUNCIÓN PARA PREDICCIÓN DE NUEVOS EMPRESARIOS
# =============================================

def predecir_actividad_probabilidades(datos_empresario):
    """
    Predice la probabilidad de cada clase de actividad para un nuevo empresario
    
    Args:
        datos_empresario: DataFrame con las características del empresario
                      (debe tener las mismas columnas que 'features')
    
    Returns:
        Diccionario con las probabilidades para cada clase
    """
    # Asegurar que estamos usando un DataFrame
    if not isinstance(datos_empresario, pd.DataFrame):
        datos_empresario = pd.DataFrame([datos_empresario])
    
    # Verificar que tenemos todas las columnas necesarias
    for col in features.columns:
        if col not in datos_empresario.columns:
            raise ValueError(f"Falta la columna {col} en los datos del empresario")
    
    # Obtener las columnas en el mismo orden que el modelo espera
    datos_input = datos_empresario[features.columns]
    
    # Obtener probabilidades
    probabilidades = model_pipeline.predict_proba(datos_input)
    
    # Crear diccionario con las probabilidades
    clases = model_pipeline.classes_
    resultado = {clase: prob[0] for clase, prob in zip(clases, probabilidades.T)}
    
    # También incluir la predicción final
    prediccion = model_pipeline.predict(datos_input)[0]
    resultado['prediccion'] = prediccion
    
    return resultado

# =============================================
# GENERACIÓN DE PERFILES IDEALES
# =============================================

def obtener_estadisticas_grupo(df, grupo):
    """
    Obtiene estadísticas descriptivas para un grupo específico
    
    Args:
        df: DataFrame con los datos
        grupo: Nombre del grupo (activo, latente, inactivo)
        
    Returns:
        Diccionario con estadísticas para cada variable
    """
    grupo_df = df[df['actividad'] == grupo]
    
    # Estadísticas para variables numéricas
    stats_num = {}
    for col in numeric_cols:
        stats_num[col] = {
            'min': grupo_df[col].min(),
            'q1': grupo_df[col].quantile(0.25),
            'median': grupo_df[col].median(),
            'q3': grupo_df[col].quantile(0.75),
            'max': grupo_df[col].max()
        }
    
    # Valores más frecuentes para variables categóricas
    stats_cat = {}
    for col in categorical_cols:
        value_counts = grupo_df[col].value_counts(normalize=True)
        stats_cat[col] = value_counts.nlargest(3).to_dict()
    
    return {'numericas': stats_num, 'categoricas': stats_cat}

# Obtener estadísticas para cada grupo
stats_activo = obtener_estadisticas_grupo(df, 'activo')
stats_latente = obtener_estadisticas_grupo(df, 'latente')
stats_inactivo = obtener_estadisticas_grupo(df, 'inactivo')

# =============================================
# FUNCIÓN PARA CREAR UN PERFIL IDEAL
# =============================================

def crear_perfil_ideal(stats):
    """
    Crea un perfil ideal consolidado basado en estadísticas del grupo
    
    Args:
        stats: Estadísticas del grupo (obtenidas con obtener_estadisticas_grupo)
        
    Returns:
        DataFrame con un perfil ideal
    """
    perfil = {}
    
    # Para variables numéricas usar la mediana (valor central)
    for col, valores in stats['numericas'].items():
        perfil[col] = valores['median']
    
    # Para variables categóricas usar el valor más frecuente
    for col, valores in stats['categoricas'].items():
        # Encontrar el valor más frecuente
        if valores:  # Verificar que hay valores
            perfil[col] = max(valores.items(), key=lambda x: x[1])[0]
        else:
            # Si no hay valores, usar un valor predeterminado
            perfil[col] = None
    
    return pd.DataFrame([perfil])

# =============================================
# FUNCIÓN PARA GENERAR MÚLTIPLES PERFILES
# =============================================

def generar_multiples_perfiles(stats, n_perfiles=5, nombre_grupo='grupo'):
    """
    Genera múltiples perfiles basados en estadísticas de un grupo
    
    Args:
        stats: Estadísticas del grupo (obtenidas con obtener_estadisticas_grupo)
        n_perfiles: Número de perfiles a generar
        nombre_grupo: Nombre del grupo para etiquetado
        
    Returns:
        DataFrame con los perfiles generados
    """
    perfiles = []
    
    for i in range(n_perfiles):
        perfil = {}
        
        # Generar variables numéricas en el rango interquartil (entre Q1 y Q3)
        for col, valores in stats['numericas'].items():
            # Usar rango interquartil para generar valores aleatorios más representativos
            perfil[col] = np.random.uniform(valores['q1'], valores['q3'])
            
            # Redondear valores según el tipo
            if col in ['edad', 'n_empleados']:
                perfil[col] = int(round(perfil[col]))
            else:
                perfil[col] = round(perfil[col], 2)
        
        # Generar variables categóricas basadas en probabilidades
        for col, valores in stats['categoricas'].items():
            # Convertir a lista para poder usar random.choices
            opciones = list(valores.keys())
            pesos = list(valores.values())
            
            # Normalizar pesos para asegurarnos de que sumen 1
            suma_pesos = sum(pesos)
            if suma_pesos > 0:  # Evitar división por cero
                pesos_normalizados = [p/suma_pesos for p in pesos]
                perfil[col] = np.random.choice(opciones, p=pesos_normalizados)
            else:
                # Si no hay pesos, elegir al azar
                perfil[col] = np.random.choice(opciones)
        
        # Añadir ID de perfil y grupo
        perfil['id'] = f"{nombre_grupo}_{i+1}"
        perfil['grupo'] = nombre_grupo
        
        perfiles.append(perfil)
    
    return pd.DataFrame(perfiles)

# Crear perfiles ideales consolidados (1 por grupo)
perfil_ideal_activo = crear_perfil_ideal(stats_activo)
perfil_ideal_latente = crear_perfil_ideal(stats_latente)
perfil_ideal_inactivo = crear_perfil_ideal(stats_inactivo)

# Añadir identificadores a los perfiles ideales
perfil_ideal_activo['id'] = 'ideal_activo'
perfil_ideal_activo['grupo'] = 'activo'
perfil_ideal_latente['id'] = 'ideal_latente'
perfil_ideal_latente['grupo'] = 'latente'
perfil_ideal_inactivo['id'] = 'ideal_inactivo'
perfil_ideal_inactivo['grupo'] = 'inactivo'

print("\n=== PERFILES IDEALES CONSOLIDADOS ===")
print("\nPerfil ACTIVO ideal:")
print(perfil_ideal_activo.to_dict('records')[0])
print("\nPerfil LATENTE ideal:")
print(perfil_ideal_latente.to_dict('records')[0])
print("\nPerfil INACTIVO ideal:")
print(perfil_ideal_inactivo.to_dict('records')[0])

# Generar múltiples perfiles por grupo (5 por grupo)
n_perfiles = 5  # Número de perfiles adicionales a generar por grupo
perfiles_activos = generar_multiples_perfiles(stats_activo, n_perfiles, 'activo')
perfiles_latentes = generar_multiples_perfiles(stats_latente, n_perfiles, 'latente')
perfiles_inactivos = generar_multiples_perfiles(stats_inactivo, n_perfiles, 'inactivo')

# Combinar todos los perfiles en una sola estructura
todos_perfiles = {
    'perfiles_ideales': {
        'activo': perfil_ideal_activo.to_dict('records')[0],
        'latente': perfil_ideal_latente.to_dict('records')[0],
        'inactivo': perfil_ideal_inactivo.to_dict('records')[0]
    },
    'perfiles_multiples': {
        'activo': perfiles_activos.to_dict('records'),
        'latente': perfiles_latentes.to_dict('records'),
        'inactivo': perfiles_inactivos.to_dict('records')
    }
}

# Guardar todos los perfiles en un solo archivo JSON
with open('perfiles_microempresarios.json', 'w', encoding='utf-8') as f:
    json.dump(todos_perfiles, f, ensure_ascii=False, indent=4)

print(f"\nSe han guardado todos los perfiles (ideales y múltiples) en 'perfiles_microempresarios.json'")

# =============================================
# PRUEBA DE PREDICCIÓN CON UN NUEVO EMPRESARIO
# =============================================

# Creamos un caso de prueba para demostrar la funcionalidad
nuevo_empresario = {
    'edad': 35,
    'nivel_educativo': 'superior', 
    'tipo_empresa': 'alimento',
    'nivel_madurez': 'media', 
    'n_empleados': 4,
    'negocio_familiar': True,
    'ingresos_semanales': 5000,
    'antiguedad': 1
}

# Convertir a DataFrame
df_nuevo = pd.DataFrame([nuevo_empresario])

# Realizar la predicción
resultado = predecir_actividad_probabilidades(df_nuevo)

print("\n=== PREDICCIÓN PARA NUEVO EMPRESARIO ===")
print(f"Datos del empresario: {nuevo_empresario}")
print("\nProbabilidades:")
for clase, prob in resultado.items():
    if clase != 'prediccion':
        print(f"- {clase}: {prob:.2%}")
print(f"\nClasificación final: {resultado['prediccion']}")

# =============================================
# FUNCIÓN PARA REENTRENAR EL MODELO EN EL FUTURO
# =============================================

def reentrenar_modelo(df_nuevos_datos, ruta_modelo='modelo_actividad_microempresarios.pkl'):
    """
    Reentrenar el modelo con nuevos datos
    
    Args:
        df_nuevos_datos: DataFrame con nuevos datos de microempresarios
        ruta_modelo: Ruta donde guardar el modelo reentrenado
        
    Returns:
        Pipeline con el modelo reentrenado
    """
    # Cargar el modelo existente
    try:
        modelo_existente = joblib.load(ruta_modelo)
        print("Modelo existente cargado correctamente")
    except:
        print("No se encontró un modelo existente. Creando uno nuevo...")
        modelo_existente = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    
    # Verificar que tenemos las columnas necesarias en los nuevos datos
    columnas_necesarias = ['edad', 'nivel_educativo', 'tipo_empresa', 'nivel_madurez', 
                          'n_empleados', 'negocio_familiar', 'ingresos_semanales', 
                          'antiguedad', 'actividad']
    
    for col in columnas_necesarias:
        if col not in df_nuevos_datos.columns:
            raise ValueError(f"Falta la columna {col} en los nuevos datos")
    
    # Preparar características y target
    X_nuevos = df_nuevos_datos[columnas_necesarias[:-1]]
    y_nuevos = df_nuevos_datos['actividad']
    
    # Reentrenar el modelo con los nuevos datos
    modelo_existente.fit(X_nuevos, y_nuevos)
    
    # Guardar el modelo reentrenado
    joblib.dump(modelo_existente, ruta_modelo)
    print(f"Modelo reentrenado guardado en {ruta_modelo}")
    
    return modelo_existente

print("\nEl código está listo para ser ejecutado. Incluye:")
print("- Entrenamiento del modelo")
print("- Generación de perfiles ideales y múltiples perfiles")
print("- Función para predecir nuevos empresarios")
print("- Función para reentrenar el modelo a futuro")