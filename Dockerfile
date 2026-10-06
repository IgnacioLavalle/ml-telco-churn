#Imagen base
FROM python:3.10-slim

#Evitar que Python escriba archivos .pyc y forzar logs inmediatos en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#Crear directorio de trabajo
WORKDIR /app

#Instalar dependencias del sistema operativo que puedan requerir Scikit-learn/LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

#Copiar solo el requirements primero 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copiar el resto del proyecto (código y modelos)
COPY src/ ./src/
COPY models/ ./models/

#Exponer el puerto de FastAPI
EXPOSE 8000

#Comando para ejecutar el servidor con Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]