# **Asistente Inteligente Avatar**

- Objetivo: Asistir en información y presentación profesional, finalmente no olvide enviar sus datos de contacto.
- Usos en frameworks de python
- (Directo) Ingresar el Link Https `http://localhost:8000/docs`.
- (Opcional) Configurar entorno `venv/`, api llm's en `.env` y Ejecutar directamente este comando `python -m ia.agent` en el directorio principal.

---
Deploy en Https:

> **Ruta: [http://localhost:8000/docs](http://localhost:8000/docs)**

### **1 Deploy Agente local**

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Debian

# Instalar dependencias
pip install -r requirements.txt


# organizar scripts a csv
python app.py

```
> **Ruta: [http://localhost:7865](http://localhost:7865)**