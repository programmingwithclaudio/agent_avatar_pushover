# **Asistente Inteligente Avatar**

- Objetivo: Asistir en información y presentación profesional, finalmente no olvide enviar sus datos de contacto.
- Usos en frameworks de python
- (Directo) Ingresar el Link Https `https://agentavatarpushover-production-62d1.up.railway.app/chat/`.
- (Opcional) Configurar entorno `venv/`, api llm's en `.env` y Ejecutar directamente este comando `python -m ia.agent` en el directorio principal.

---
Deploy en Https:
> **Agente IA Avatar: [https://agentavatarpushover-production-62d1.up.railway.app/chat/](https://agentavatarpushover-production-62d1.up.railway.app/chat/)**

> **Ruta Enpoints: [https://agentavatarpushover-production-62d1.up.railway.app/docs](https://agentavatarpushover-production-62d1.up.railway.app/docs)**

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
python agent.py

```
> **Ruta: [http://localhost:7865](http://localhost:7865)**