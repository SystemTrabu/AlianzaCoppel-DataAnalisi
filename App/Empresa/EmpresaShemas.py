from App.models import ma

from App.models.EmpresaModel import Empresa

class EmpresaSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Empresa
        load_instance = True
        include_fk = True

Empresa_schema = EmpresaSchema()
Empresas_schema = EmpresaSchema(many=True)
