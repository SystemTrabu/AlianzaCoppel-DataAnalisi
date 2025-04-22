from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from .ColaboradoresModel import Usuario
from .MicroEmpresariosModel import MicroEmpresario
from .CategoriaModel import Categoria
from .CursosModel import Curso
from .CursosTerminadosModel import CursosTerminados
from .EmpresaModel import Empresa