from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# models/__init__.py o donde agrupes los modelos

from .ColaboradoresModel import Usuario
from .MicroEmpresariosModel import MicroEmpresario
from .CategoriaModel import Categoria
from .CursosModel import Curso               # Primero importa Curso
from .CursosTerminadosModel import CursosTerminados  # Luego CursosTerminados
from .EmpresaModel import Empresa
