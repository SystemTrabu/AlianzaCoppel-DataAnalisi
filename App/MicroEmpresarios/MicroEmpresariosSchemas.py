from App.models import ma , MicroEmpresario

from App.models.MicroEmpresariosModel import MicroEmpresario

class MicroEmpresaioSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = MicroEmpresario
        load_instance = True
        include_fk = True

MicroEmpresario_schema = MicroEmpresaioSchema()
MicroEmpresarios_schema = MicroEmpresaioSchema(many=True)
