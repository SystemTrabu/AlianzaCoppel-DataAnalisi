# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config
from flasgger import Swagger
from .models import db, ma  

migrate = Migrate()

def create_app():
    app = Flask(__name__)

    app.config.from_object(Config)

    db.init_app(app)
    ma.init_app(app)  
    migrate.init_app(app, db)

    # Importa los modelos para que Flask-Migrate los detecte
    
    swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # incluye todas las rutas
            "model_filter": lambda tag: True,  # incluye todos los modelos
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
    "definitions": {
        "MicroEmpresarioGET": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "nombre_empresario": {"type": "string"},
                "correo": {"type": "string"},
                "genero": {"type": "string"},
                "n_telefono": {"type": "string"},
                "codigo_postal": {"type": "integer"},
                "Webinars": {"type": "integer"},
                "fecha_registro": {"type": "string", "format": "date-time"},
                "fecha_nacimiento": {"type": "string", "format": "date-time"},
                "nivel_educativo": {"type": "string"},
                "actividad": {"type": ["string", "null"]},
                "estado": {"type": "string"},
                "colaborador_id": {"type": "integer"},
                "empresa_id": {"type": "integer"}
            }
        },
        "RegistroCompletoPOST": {
            "type": "object",
            "properties": {
                "nombre_empresa": {"type": "string"},
                "tipo_empresa": {"type": "string"},
                "n_empleados": {"type": "integer"},
                "ingresos_semanales": {"type": "integer"},
                "nivel_madurez": {"type": "string"},
                "negocio_familiar": {"type": "boolean"},
                "antiguedad": {"type": "integer"},
                "estado": {"type": "string"},

                "nombre_empresario": {"type": "string"},
                "genero": {"type": "string"},
                "correo": {"type": "string"},
                "fecha_nacimiento": {"type": "string", "format": "date-time"},
                "fecha_registro": {"type": "string", "format": "date-time"},
                "n_telefono": {"type": "string"},
                "codigo_postal": {"type": "integer"},
                "Webinars": {"type": "integer"},
                "nivel_educativo": {"type": "string"},
                "colaborador_id": {"type": "integer"}
            }
        },
        "empresas": {
            "type": "object",
            "properties": {
                "antiguedad": {"type": "integer"},
                "id": {"type": "integer"},
                "ingresos_semanales": {"type": "integer"},
                "n_empleados": {"type": "integer"},
                "negocio_familiar": {"type": "boolean"},
                "nivel_madurez": {"type": "string"},
                "nombre_empresa": {"type": "string"},
                "tipo_empresa": {"type": "string"}
            }
            },
        "colaboradoresGetEmpresarios": {
            "type": "object",
            "properties": {
                "CP": { "type": "integer" },
                "Colaborador_ID": { "type": "integer" },
                "Cursos_terminados": { "type": "integer" },
                "Empresa_ID": { "type": "integer" },
                "Fecha_Nacimiento": { "type": "string", "format": "date-time" },
                "Fecha_Registro": { "type": "string", "format": "date-time" },
                "Nivel_Educativo": { "type": "string" },
                "Nombre_empresa": { "type": "string" },
                "Num_tel": { "type": "string" },
                "Webinars": { "type": "integer" },
                "correo": { "type": "string" },
                "genero": { "type": "string" },
                "id": { "type": "integer" },
                "nombre": { "type": "string" }
                }
            },
        "colaboradores": {
            "type": "object",
            "properties": {
                "Num_Empleado": { "type": "integer" },
                "id": { "type": "integer" },
                "nombre": { "type": "string" }
}
            },
        "colaboradoresGet": {
            "type": "object",
            "properties": {
                "existe": { "type": "boolean" },
                "id": { "type": "integer" },
                "nombre": { "type": "string" }

}
            },
         "colaboradoresGet": {
            "type": "object",
            "properties": {
                "existe": { "type": "boolean" },
                "id": { "type": "integer" },
                "nombre": { "type": "string" }

}
            }
        
        }
    }

    swagger = Swagger(app, config=swagger_config)

    from . import models

    # Registra los blueprints al final para evitar imports circulares
    from .AnaisisDatos.AnalisisControlls import analisis_bp
    from .Colaboradores.ColaboradorController import colaborador_bp
    app.register_blueprint(analisis_bp)

    from .MicroEmpresarios.MicroEmpresariosControlls import MicroEmpresario_bp
    app.register_blueprint(MicroEmpresario_bp)
    
    from .Empresa.EmpresaControlls import Empresas_bp
    app.register_blueprint(Empresas_bp)

    from.Colaboradores.ColaboradorController import colaborador_bp
    app.register_blueprint(colaborador_bp)

    return app

