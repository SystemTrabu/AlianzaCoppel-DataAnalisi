# AlianzaCoppel-DataAnalisis

Este proyecto implementa una API REST usando **Flask** y documentada con **Flasgger** (Swagger UI), diseñada para  los colaboradores de coppel con el objetivo de un inpacto social 

## 🚀 Tecnologías colab

- Python 3.x
- Flask
- Flasgger
- Pandas
- JSON / CSV

## ⚙️ Instalación

```bash

# Crear entorno virtual
python -m venv venv
source venv/bin/activate     # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt


Ejecución del servidor

La API estará disponible en: http://localhost:5000

La documentación Swagger estará en: http://localhost:5000/apidocs/#/

# Crear entorno 
 📌 Endpoints disponibles

GET  ​
/api​/analisis​/colaboradores​/efectividad
Genera un ranking de los colaboradores

GET
/api​/analisis​/getEmpresarios
Obtener todos los empresarios que son activos, inactivos y latentes

GET
​/api​/analisis​/mejorefectividad
Mejor colaborador top 1

GET
​/api​/analisis​/generar​/reporte
Genera un reporte con diagramas y datos

GET
​/api​/colaborador​/getColaboradores
Obtener todos los colaboradores

GET
​/api​/colaborador​/getEmpresarios​/{id}
Obtener todos los empresarios con la id del colaborador

GET
​/api​/colaborador​/verificar​/{id}
verificar colaborador atravez del n de control

GET
​/api​/empresa​/
Obtener todas las empresas

POST
​/api​/empresa​/
Crear nueva empresa

DELETE
​/api​/empresa​/{id}
Eliminar una por ID

GET
​/api​/empresa​/{id}
Obtener un microempresario por ID

GET
​/api​/empresarios​/
Obtener todos los microempresarios

POST
​/api​/empresarios​/
Crear un nuevo microempresario

DELETE
​/api​/empresarios​/{id}
Eliminar un microempresario por ID

GET
​/api​/empresarios​/{id}