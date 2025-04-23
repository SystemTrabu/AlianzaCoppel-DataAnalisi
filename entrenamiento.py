import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine


engine = create_engine('postgresql://postgres:HRxlTXYjtnYgTVbehhRqwnCEnopFFegE@shuttle.proxy.rlwy.net:40252/railway')

# Extraer datos de las tablas
query_usuarios = "SELECT * FROM microempresario"
query_empresas = "SELECT * FROM empresas" 
query_cursos = 'SELECT * FROM "CursosTerminados"'

usuarios_df = pd.read_sql(query_usuarios, engine)
empresas_df = pd.read_sql(query_empresas, engine)
cursos_df = pd.read_sql(query_cursos, engine)



df_combinado = pd.merge(usuarios_df, empresas_df, left_on='empresa_id', right_on='id', how='left')

# Preparar y limpiar datos
# Calcular edad

df_combinado['edad'] = pd.to_datetime('today').year - pd.to_datetime(df_combinado['fecha_nacimiento']).dt.year

nivel_madurez_map = {
    'bajo': 1, 
    'medio': 2, 
    'alto': 3
}
df_combinado['nivel_madurez_num'] = df_combinado['nivel_madurez'].map(nivel_madurez_map)

# Codificar variables categóricas
df_combinado = pd.get_dummies(df_combinado, columns=['genero', 'tipo_empresa'], drop_first=False)

# Asegurar que todas las columnas estén presentes aunque no existan en el set actual
for col in ['genero_hombre', 'genero_mujer', 'genero_otro']:
    if col not in df_combinado.columns:
        df_combinado[col] = 0


# Calcular días desde registro
df_combinado['dias_desde_registro'] = (pd.to_datetime('today') - 
                                       pd.to_datetime(df_combinado['fecha_registro'])).dt.days

# Calcular métricas de actividad
fecha_limite = datetime.now() - timedelta(days=90)
cursos_recientes = cursos_df[pd.to_datetime(cursos_df['fecha']) >= fecha_limite]

# Contar cursos por usuario en los últimos 3 meses
cursos_por_usuario = cursos_recientes.groupby('microempresario_id').size().reset_index(name='cursos_ultimos_3meses')
df_combinado = pd.merge(df_combinado, cursos_por_usuario, left_on='id_x', right_on='microempresario_id', how='left')
df_combinado['cursos_ultimos_3meses'] = df_combinado['cursos_ultimos_3meses'].fillna(0)
print(df_combinado.columns.tolist())
print(df_combinado.head())
# Clasificar a los usuarios según su actividad
df_combinado['estado_actividad'] = pd.cut(
    df_combinado['cursos_ultimos_3meses'],
    bins=[-1, 5, 15, float('inf')],
    labels=['inactivo', 'latente', 'activo']
)


# Análisis exploratorio básico
print(df_combinado['estado_actividad'].value_counts())

# Visualizar distribución de actividad por nivel educativo
plt.figure(figsize=(12, 6))
sns.countplot(x='nivel_educativo', hue='estado_actividad', data=df_combinado)
plt.title('Distribución de actividad por nivel educativo')
plt.xticks(rotation=45)
plt.show()

# Visualizar distribución de actividad por edad
plt.figure(figsize=(12, 6))
sns.boxplot(x='estado_actividad', y='edad', data=df_combinado)
plt.title('Distribución de edad por estado de actividad')
plt.show()

# Visualizar distribución de actividad por nivel de madurez del negocio
plt.figure(figsize=(12, 6))
sns.countplot(x='nivel_madurez_num', hue='estado_actividad', data=df_combinado)
plt.title('Distribución de actividad por nivel de madurez')
plt.xticks([0, 1, 2], ['Idea de negocio', 'Recién iniciado', '3+ años'])
plt.show()


# Seleccionar características para clustering
features_clustering = ['edad', 'nivel_madurez_num', 'n_empleados', 
                       'ingresos_semanales', 'dias_desde_registro', 
                       'cursos_ultimos_3meses']

print(df_combinado.columns)
cluster_data = df_combinado[features_clustering].copy()
cluster_data = cluster_data.fillna(cluster_data.mean())  # Manejar valores faltantes

# Escalar las características
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Determinar el número óptimo de clusters usando el método del codo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Visualizar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del codo para determinar número óptimo de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Basado en el gráfico, elegimos un número óptimo de clusters (por ejemplo, 4)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_combinado['cluster'] = kmeans.fit_predict(scaled_data)


cluster_profiles = df_combinado.groupby('cluster').agg({
    'edad': 'mean',
    'nivel_madurez_num': 'mean',
    'n_empleados': 'mean',
    'ingresos_semanales': 'mean',
    'cursos_ultimos_3meses': 'mean',
    'estado_actividad': lambda x: x.value_counts().index[0] 
}).reset_index()

print("Perfiles de microempresarios:")
print(cluster_profiles)

# Visualizar la relación entre clusters y estado de actividad
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='estado_actividad', data=df_combinado)
plt.title('Distribución de estados de actividad por cluster')
plt.show()



# Preparar datos para el modelo predictivo
X = df_combinado[['edad', 'nivel_madurez_num', 'n_empleados', 
                 'ingresos_semanales', 'dias_desde_registro',
                 'genero_hombre', 'genero_mujer', 'genero_otro']]

y = df_combinado['estado_actividad']

# Manejar valores faltantes
X = X.fillna(X.mean())

# Dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Entrenar un modelo RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluar el modelo
from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualizar importancia de características
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importances)
plt.title('Importancia de las características para predecir estado de actividad')
plt.show()


# Analizar probabilidades para cada clase
y_probs = clf.predict_proba(X_test)
prob_df = pd.DataFrame(y_probs, columns=clf.classes_)

# Función para predecir probabilidad para un nuevo microempresario
def predecir_perfil(edad, nivel_madurez, empleados, ingresos, dias_registro, genero):
    # Crear un array con los datos del nuevo microempresario
    genero_hombre = 1 if genero == 'hombre' else 0
    genero_mujer = 1 if genero == 'mujer' else 0
    genero_otro = 1 if genero == 'otro' else 0

    nuevo = np.array([[edad, nivel_madurez, empleados, ingresos, dias_registro, 
                    genero_hombre, genero_mujer, genero_otro]])

    
    # Obtener probabilidades para cada clase
    probs = clf.predict_proba(nuevo)[0]
    
    # Crear un diccionario con las probabilidades
    resultado = {clase: prob for clase, prob in zip(clf.classes_, probs)}
    
    # Determinar el cluster al que pertenece
    nuevo_scaled = scaler.transform(nuevo[:, :len(features_clustering)])
    cluster = kmeans.predict(nuevo_scaled)[0]
    
    return {
        'probabilidades': resultado,
        'cluster': cluster,
        'perfil': cluster_profiles[cluster_profiles['cluster'] == cluster].to_dict('records')[0]
    }

# Ejemplo de uso
ejemplo = predecir_perfil(
    edad=35, 
    nivel_madurez=2, 
    empleados=3, 
    ingresos=5000, 
    dias_registro=60,
    genero='mujer'
)
print("Predicción para nuevo microempresario:")
print(ejemplo)


import joblib

# Guardar el modelo de clasificación
joblib.dump(clf, 'modelo_prediccion_microempresarios.pkl')

# Guardar el modelo de clustering y el scaler
joblib.dump(kmeans, 'modelo_clustering_microempresarios.pkl')
joblib.dump(scaler, 'scaler_microempresarios.pkl')

print("Modelos guardados correctamente")