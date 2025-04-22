# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config
from .models import db  # db se importa de models/__init__.py

migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app, db)

    # Importa los modelos para que Flask-Migrate los detecte
    from . import models

    from .AnaisisDatos.AnalisisControlls import analisis_bp
    app.register_blueprint(analisis_bp)
    return app
