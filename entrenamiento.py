import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import joblib

# Configuración de visualización
plt.style.use('ggplot')
sns.set_palette("husl")

# 1. Conexión a la base de datos y extracción de datos
engine = create_engine('postgresql://postgres:HRxlTXYjtnYgTVbehhRqwnCEnopFFegE@shuttle.proxy.rlwy.net:40252/railway')

# Cargar datos
df_empresarios = pd.read_sql("SELECT * FROM microempresario", engine)
df_empresas = pd.read_sql("SELECT * FROM empresas", engine)
df_cursos = pd.read_sql('SELECT * FROM "CursosTerminados"', engine)

# 2. Procesamiento de datos
# Calcular cantidad de cursos por microempresario
cursos_count = df_cursos.groupby('microempresario_id').size().reset_index(name='cursos_completados')

# Unir datos
df = pd.merge(df_empresarios, cursos_count, left_on='id', right_on='microempresario_id', how='left')
df = pd.merge(df, df_empresas, left_on='empresa_id', right_on='id', how='left')

# Rellenar NaN con 0 (para quienes no han completado cursos)
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

# 3. Ingresos semanales por actividad
plt.figure(figsize=(12, 6))
sns.boxplot(x='actividad', y='ingresos_semanales', data=df, order=['activo', 'latente', 'inactivo'])
plt.title('Distribución de Ingresos Semanales por Nivel de Actividad')
plt.xlabel('Nivel de Actividad')
plt.ylabel('Ingresos Semanales')
plt.tight_layout()
plt.show()

# 4. Participación en webinars
plt.figure(figsize=(12, 6))
sns.countplot(x='Webinars', hue='actividad', data=df, 
              hue_order=['activo', 'latente', 'inactivo'],
              palette="Set2")
plt.title('Participación en Webinars por Nivel de Actividad')
plt.xlabel('Número de Webinars Asistidos')
plt.ylabel('Cantidad de Microempresarios')
plt.legend(title='Nivel de Actividad')
plt.tight_layout()
plt.show()

# 5. Tipo de empresa por actividad
plt.figure(figsize=(14, 7))
df_temp = df.groupby(['tipo_empresa', 'actividad']).size().unstack()
df_temp.plot(kind='bar', stacked=True)
plt.title('Distribución de Tipos de Empresa por Nivel de Actividad')
plt.xlabel('Tipo de Empresa')
plt.ylabel('Cantidad de Microempresarios')
plt.legend(title='Nivel de Actividad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Nivel educativo por actividad
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
plt.show()

# =============================================
# MODELADO (continuación del código original)
# =============================================

# 3. Preparación de características
features = df[[
    'edad', 'genero', 'nivel_educativo', 'Webinars',
    'tipo_empresa', 'nivel_madurez', 'n_empleados', 
    'negocio_familiar', 'ingresos_semanales', 'antiguedad'
]]

target = df['actividad']

# 4. Preprocesamiento
categorical_cols = ['genero', 'nivel_educativo', 'tipo_empresa', 'nivel_madurez', 'negocio_familiar']
numeric_cols = ['edad', 'Webinars', 'n_empleados', 'ingresos_semanales', 'antiguedad']

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

# 5. Creación y entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# 6. Evaluación
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Guardar modelo
joblib.dump(model, 'modelo_actividad_microempresarios.pkl')

# 8. Función para predecir nuevos casos
def predecir_actividad(nuevos_datos):
    """
    Función para predecir actividad de nuevos microempresarios
    
    Args:
        nuevos_datos: DataFrame con las mismas columnas que features
        
    Returns:
        Predicciones de actividad (activo, latente, inactivo)
    """
    return model.predict(nuevos_datos)

# 9. Ejemplo de perfil ideal
perfil_activo = pd.DataFrame({
    'edad': [35],
    'genero': ['Masculino'],
    'nivel_educativo': ['Licenciatura'],
    'Webinars': [5],
    'tipo_empresa': ['Tecnología'],
    'nivel_madurez': ['En crecimiento'],
    'n_empleados': [5],
    'negocio_familiar': [False],
    'ingresos_semanales': [15000],
    'antiguedad': [4]
})

perfil_latente = pd.DataFrame({
    'edad': [45],
    'genero': ['Femenino'],
    'nivel_educativo': ['Preparatoria'],
    'Webinars': [2],
    'tipo_empresa': ['Comercio'],
    'nivel_madurez': ['Iniciando'],
    'n_empleados': [3],
    'negocio_familiar': [True],
    'ingresos_semanales': [8000],
    'antiguedad': [2]
})

perfil_inactivo = pd.DataFrame({
    'edad': [50],
    'genero': ['Masculino'],
    'nivel_educativo': ['Secundaria'],
    'Webinars': [0],
    'tipo_empresa': ['Manufactura'],
    'nivel_madurez': ['Establecido'],
    'n_empleados': [1],
    'negocio_familiar': [True],
    'ingresos_semanales': [5000],
    'antiguedad': [10]
})

# Predecir para verificar que coinciden
print("\nPredicción perfil activo:", predecir_actividad(perfil_activo)[0])
print("Predicción perfil latente:", predecir_actividad(perfil_latente)[0])
print("Predicción perfil inactivo:", predecir_actividad(perfil_inactivo)[0])

# Visualización adicional: Importancia de características
# Extraer el modelo entrenado
rf_model = model.named_steps['classifier']

# Obtener nombres de características después del preprocesamiento
feature_names = (numeric_cols + 
                 list(model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_cols)))

# Crear DataFrame de importancia
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Gráfico de importancia de características
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Importancia de Características en el Modelo')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.tight_layout()
plt.show()