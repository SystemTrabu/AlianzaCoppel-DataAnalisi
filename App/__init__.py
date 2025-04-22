# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config
from .models import db, ma  

migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    ma.init_app(app)  
    migrate.init_app(app, db)

    # Importa los modelos para que Flask-Migrate los detecte
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

