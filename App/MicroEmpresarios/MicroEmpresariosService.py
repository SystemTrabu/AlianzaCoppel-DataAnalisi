from datetime import date
import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from .MicroEmpresariosRepository import MicroEmpresarioRepository
from App.models import db
from App.models.MicroEmpresariosModel import MicroEmpresario
from App.models.EmpresaModel import Empresa

class MicroEmpresariosService:

    @staticmethod
    def listar():
        return  MicroEmpresarioRepository.get_all()

    @staticmethod
    def obtener(id):
        return  MicroEmpresarioRepository.get_by_id(id)

    
    @staticmethod
    def eliminar(id):
        return  MicroEmpresarioRepository.delete(id)

    def crear(data):
        # 1. Crear primero la empresa (negocio)
        negocio = Empresa(
            nombre_empresa=data['nombre_empresa'],
            tipo_empresa=data['tipo_empresa'],
            n_empleados=data.get('n_empleados'),
            ingresos_semanales=data.get('ingresos_semanales'),
            nivel_madurez=data.get('nivel_madurez'),
            negocio_familiar=data.get('negocio_familiar'),
            antiguedad=data.get('antiguedad')
        )
        db.session.add(negocio)
        db.session.flush()  # Para obtener el ID de empresa

        # 2. Crear MicroEmpresario con empresa_id generado
        empresario = MicroEmpresario(
            nombre_empresario=data['nombre_empresario'],
            genero=data.get('genero'),
            correo=data.get('correo'),
            fecha_nacimiento=data.get('fecha_nacimiento'),
            fecha_registro=data.get('fecha_registro'),
            n_telefono=data['n_telefono'],
            codigo_postal=data['codigo_postal'],
            Webinars=data['Webinars'],
            nivel_educativo=data.get('nivel_educativo'),
            colaborador_id=data.get('colaborador_id'),
            estado=data.get('estado'),
            empresa_id=negocio.id  # ID generado en flush
        )
        db.session.add(empresario)
        db.session.commit()

        return empresario, negocio
    
    def predecir_actividad(empresario, empresa):
        import os
        import pickle
        import pandas as pd
        import joblib
        from datetime import date
        from sklearn.ensemble import RandomForestClassifier
        
        try:
            ruta_actual = os.path.dirname(__file__)
            ruta = os.path.abspath(os.path.join(ruta_actual, "../../modelo_actividad_microempresarios.pkl"))
            
            if not os.path.exists(ruta):
                return {"error": "El modelo no existe en la ruta especificada"}
            
            try:
                modelo = joblib.load(ruta)
            except:
                with open(ruta, "rb") as f:
                    modelo = pickle.load(f)
            
            def calcular_edad(fecha_nacimiento):
                hoy = date.today()
                return hoy.year - fecha_nacimiento.year - ((hoy.month, hoy.day) < (fecha_nacimiento.month, fecha_nacimiento.day))
            
            nuevo_empresario = {
                'edad': calcular_edad(empresario.fecha_nacimiento),
                'nivel_educativo': empresario.nivel_educativo,
                'tipo_empresa': empresa.tipo_empresa,
                'nivel_madurez': empresa.nivel_madurez,
                'n_empleados': empresa.n_empleados,
                'negocio_familiar': empresa.negocio_familiar,
                'ingresos_semanales': empresa.ingresos_semanales,
                'antiguedad': empresa.antiguedad
            }
            
            df_input = pd.DataFrame([nuevo_empresario])
            
            if hasattr(modelo, 'predict_proba'):
                probabilidades = modelo.predict_proba(df_input)
                clases = modelo.classes_
                
                resultado = {clase: float(prob) for clase, prob in zip(clases, probabilidades[0])}
                
                prediccion = modelo.predict(df_input)[0]
                resultado['prediccion'] = prediccion
            else:
                prediccion = modelo.predict(df_input)[0]
                resultado = {
                    'prediccion': prediccion,
                    'activo': 1.0 if prediccion == 'activo' else 0.0,
                    'latente': 1.0 if prediccion == 'latente' else 0.0,
                    'inactivo': 1.0 if prediccion == 'inactivo' else 0.0
                }
            
          
            return resultado
            
        except Exception as e:
            print(f"Error al predecir actividad: {str(e)}")
            return {"error": str(e)}