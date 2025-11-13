from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import pandas as pd

load_dotenv(override=True)

# ========================================
# 📦 CONFIGURACIÓN CENTRALIZADA
# ========================================
class Config:
    """Configuración centralizada de la aplicación"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
    PUSHOVER_USER = os.getenv("PUSHOVER_USER")
    MODEL = "gpt-4o-mini"
    
    # Rutas de archivos
    LINKEDIN_PDF = "me/linkedin.pdf"
    SUMMARY_TXT = "me/summary.txt"
    PROJECTS_CSV = "datasets/resumen/repos_con_tags_dinamicos.csv"
    METADATA_JSON = "datasets/resumen/metadata_dinamica.json"


# ========================================
# 🔔 SERVICIO DE NOTIFICACIONES
# ========================================
class NotificationService:
    """Maneja envío de notificaciones por Pushover"""
    
    @staticmethod
    def send(message):
        """Envía notificación push"""
        try:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": Config.PUSHOVER_TOKEN,
                    "user": Config.PUSHOVER_USER,
                    "message": message,
                }
            )
        except Exception as e:
            print(f"Error enviando notificación: {e}")


# ========================================
# 📂 CARGADOR DE DATOS DEL PERFIL
# ========================================
class ProfileLoader:
    """Carga y gestiona datos del perfil profesional"""
    
    def __init__(self, name="Claudio Quispe Alarcon"):
        self.name = name
        self.linkedin = self._load_linkedin()
        self.summary = self._load_summary()
    
    def _load_linkedin(self):
        """Extrae texto del PDF de LinkedIn"""
        try:
            reader = PdfReader(Config.LINKEDIN_PDF)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except FileNotFoundError:
            print(f"⚠️ No se encontró {Config.LINKEDIN_PDF}")
            return "Perfil de LinkedIn no disponible"
    
    def _load_summary(self):
        """Lee el resumen profesional"""
        try:
            with open(Config.SUMMARY_TXT, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ No se encontró {Config.SUMMARY_TXT}")
            return "Resumen profesional no disponible"


# ========================================
# 🔍 REPOSITORIO DE PROYECTOS
# ========================================
class ProjectRepository:
    """Maneja búsqueda y consulta de proyectos"""
    
    def __init__(self):
        self.projects_df = self._load_projects()
        self.metadata = self._load_metadata()
    
    def _load_projects(self):
        """Carga DataFrame de proyectos"""
        try:
            return pd.read_csv(Config.PROJECTS_CSV, sep=",")
        except FileNotFoundError:
            print(f"⚠️ No se encontró {Config.PROJECTS_CSV}")
            return pd.DataFrame()
    
    def _load_metadata(self):
        """Carga metadata de proyectos"""
        try:
            with open(Config.METADATA_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ No se encontró {Config.METADATA_JSON}")
            return {}
    
    def search(self, dominio=None, tecnologia=None, tipo_proyecto=None, incluye_ml=False, limit=5):
        """Busca proyectos según criterios"""
        if self.projects_df.empty:
            return {"error": "No hay proyectos disponibles"}
        
        proyectos_encontrados = []
        
        for idx, row in self.projects_df.iterrows():
            try:
                clasificacion = json.loads(row['clasificacion_dinamica'])
                
                # Aplicar filtros
                if not self._match_filters(clasificacion, dominio, tecnologia, tipo_proyecto, incluye_ml):
                    continue
                
                # Construir información del proyecto
                proyecto_info = {
                    'nombre': row.get('url_repositorio', '').split('/')[-1],
                    'url': row.get('url_repositorio', 'N/A'),
                    'proposito': clasificacion.get('proposito_principal', 'Sin descripción'),
                    'dominio': clasificacion.get('dominio_aplicacion', 'N/A'),
                    'tipo': ', '.join(clasificacion.get('tipo_proyecto', [])),
                    'tecnologias_backend': ', '.join(clasificacion.get('tecnologias_backend', [])),
                    'tecnologias_frontend': ', '.join(clasificacion.get('tecnologias_frontend', [])),
                    'bases_datos': ', '.join(clasificacion.get('bases_datos', [])),
                    'ml_ia': ', '.join(clasificacion.get('ml_ia', [])),
                    'funcionalidades': ', '.join(clasificacion.get('funcionalidades_clave', [])[:3]),
                }
                
                proyectos_encontrados.append(proyecto_info)
                
                if len(proyectos_encontrados) >= limit:
                    break
                    
            except json.JSONDecodeError:
                continue
        
        return {
            "encontrados": len(proyectos_encontrados),
            "proyectos": proyectos_encontrados,
            "total_portafolio": len(self.projects_df)
        }
    
    def _match_filters(self, clasificacion, dominio, tecnologia, tipo_proyecto, incluye_ml):
        """Verifica si un proyecto cumple con los filtros"""
        # Filtro de dominio
        if dominio and clasificacion.get('dominio_aplicacion', '').lower() != dominio.lower():
            return False
        
        # Filtro de tecnología
        if tecnologia:
            todas_tech = (
                clasificacion.get('tecnologias_backend', []) +
                clasificacion.get('tecnologias_frontend', []) +
                clasificacion.get('bases_datos', []) +
                clasificacion.get('devops_cloud', [])
            )
            if not any(tecnologia.lower() in tech.lower() for tech in todas_tech):
                return False
        
        # Filtro de tipo de proyecto
        if tipo_proyecto:
            if not any(tipo_proyecto.lower() in tipo.lower() for tipo in clasificacion.get('tipo_proyecto', [])):
                return False
        
        # Filtro ML/IA
        if incluye_ml and not clasificacion.get('ml_ia', []):
            return False
        
        return True
    
    def get_expertise(self, categoria="general"):
        """Retorna expertise técnico según categoría"""
        if not self.metadata:
            return {"error": "Metadata no disponible"}
        
        if categoria == "general":
            return {
                "total_proyectos": self.metadata.get('total_proyectos', 0),
                "estadisticas_generales": self.metadata.get('estadisticas', {}),
                "dominios_principales": dict(list(self.metadata.get('dominios_aplicacion', {}).items())[:10]),
                "top_backend": dict(list(self.metadata.get('top_tecnologias', {}).get('backend', {}).items())[:10]),
                "top_frontend": dict(list(self.metadata.get('top_tecnologias', {}).get('frontend', {}).items())[:5]),
                "top_ml_ia": dict(list(self.metadata.get('top_tecnologias', {}).get('ml_ia', {}).items())[:10]),
            }
        
        elif categoria == "backend":
            stats = self.metadata.get('estadisticas', {})
            total = self.metadata.get('total_proyectos', 1)
            return {
                "tecnologias": self.metadata.get('top_tecnologias', {}).get('backend', {}),
                "bases_datos": self.metadata.get('top_tecnologias', {}).get('bases_datos', {}),
                "proyectos_backend": stats.get('proyectos_con_backend', 0),
                "porcentaje": f"{(stats.get('proyectos_con_backend', 0) / total * 100):.1f}%"
            }
        
        elif categoria == "frontend":
            stats = self.metadata.get('estadisticas', {})
            total = self.metadata.get('total_proyectos', 1)
            return {
                "tecnologias": self.metadata.get('top_tecnologias', {}).get('frontend', {}),
                "proyectos_frontend": stats.get('proyectos_con_frontend', 0),
                "porcentaje": f"{(stats.get('proyectos_con_frontend', 0) / total * 100):.1f}%"
            }
        
        elif categoria in ["ml", "ia"]:
            stats = self.metadata.get('estadisticas', {})
            total = self.metadata.get('total_proyectos', 1)
            return {
                "tecnologias_ml_ia": self.metadata.get('top_tecnologias', {}).get('ml_ia', {}),
                "proyectos_ml_ia": stats.get('proyectos_con_ml_ia', 0),
                "porcentaje": f"{(stats.get('proyectos_con_ml_ia', 0) / total * 100):.1f}%"
            }
        
        else:
            return {"error": f"Categoría '{categoria}' no reconocida"}


# ========================================
# 🛠️ HERRAMIENTAS DISPONIBLES
# ========================================
def record_user_details(email, name="Nombre no indicado", notes="no proporcionadas"):
    """Registra detalles de contacto del usuario"""
    NotificationService.send(f"📧 Contacto: {name} | Email: {email} | Notas: {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    """Registra pregunta que no se pudo responder"""
    NotificationService.send(f"❓ Pregunta sin respuesta: {question}")
    return {"recorded": "ok"}

def search_projects(dominio=None, tecnologia=None, tipo_proyecto=None, incluye_ml=False, limit=5):
    """Busca proyectos en el portfolio"""
    repo = ProjectRepository()
    return repo.search(dominio, tecnologia, tipo_proyecto, incluye_ml, limit)

def get_technical_expertise(categoria="general"):
    """Obtiene expertise técnico"""
    repo = ProjectRepository()
    return repo.get_expertise(categoria)


# ========================================
# 📋 DEFINICIONES JSON DE HERRAMIENTAS
# ========================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Registra información de contacto del usuario interesado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Email del usuario"},
                    "name": {"type": "string", "description": "Nombre del usuario"},
                    "notes": {"type": "string", "description": "Notas adicionales del contexto"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Registra preguntas que no se pudieron responder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Pregunta sin respuesta"}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_projects",
            "description": "Busca proyectos específicos por dominio, tecnología o tipo. Úsala para '¿Qué proyectos has hecho con FastAPI?' o '¿Tienes experiencia en E-commerce?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "dominio": {"type": "string", "description": "Dominio (E-commerce, Finanzas, ML, etc.)"},
                    "tecnologia": {"type": "string", "description": "Tecnología (FastAPI, React, PostgreSQL, etc.)"},
                    "tipo_proyecto": {"type": "string", "description": "Tipo (API REST, Dashboard, Bot, etc.)"},
                    "incluye_ml": {"type": "boolean", "description": "Filtrar solo proyectos ML/IA"},
                    "limit": {"type": "integer", "description": "Máximo de proyectos a retornar"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_expertise",
            "description": "Muestra expertise técnico y estadísticas del portfolio. Úsala para '¿Cuál es tu stack?' o '¿Cuántos proyectos de ML has hecho?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "categoria": {
                        "type": "string",
                        "description": "Categoría: general, backend, frontend, ml, ia, devops, fullstack, dominios",
                        "enum": ["general", "backend", "frontend", "ml", "ia", "devops", "fullstack", "dominios"]
                    }
                },
                "required": []
            }
        }
    }
]


# ========================================
# 💬 GESTOR DE CHAT
# ========================================
class ChatManager:
    """Maneja la lógica del chat con OpenAI"""
    
    def __init__(self, profile: ProfileLoader):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.profile = profile
    
    def build_system_prompt(self):
        """Construye el prompt del sistema"""
        return f"""Actúas como {self.profile.name}. Respondes preguntas en su sitio web sobre su trayectoria profesional, habilidades y experiencia.

Tu responsabilidad es representar a {self.profile.name} con fidelidad, usando un tono profesional y cercano.

IMPORTANTE: Tienes acceso a herramientas para:
- Buscar proyectos específicos (search_projects)
- Mostrar expertise técnico (get_technical_expertise)
- Registrar contactos (record_user_details)
- Registrar preguntas sin respuesta (record_unknown_question)

Si no sabes algo, usa 'record_unknown_question'.
Si el usuario muestra interés, pide su email y usa 'record_user_details'.

## Resumen:
{self.profile.summary}

## Perfil de LinkedIn:
{self.profile.linkedin}

Mantente siempre en el personaje de {self.profile.name}."""
    
    def chat(self, message, history):
        """Procesa un mensaje del chat"""
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ] + history + [
            {"role": "user", "content": message}
        ]
        
        # Loop para manejar tool calls
        while True:
            response = self.client.chat.completions.create(
                model=Config.MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA
            )
            
            choice = response.choices[0]
            
            if choice.finish_reason == "tool_calls":
                # Procesar llamadas a herramientas
                messages.append(choice.message)
                tool_results = self._execute_tools(choice.message.tool_calls)
                messages.extend(tool_results)
            else:
                # Respuesta final
                return choice.message.content
    
    def _execute_tools(self, tool_calls):
        """Ejecuta las herramientas solicitadas"""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"🔧 Ejecutando: {tool_name} con {arguments}", flush=True)
            
            # Ejecutar la función correspondiente
            tool_function = globals().get(tool_name)
            if tool_function:
                result = tool_function(**arguments)
            else:
                result = {"error": f"Herramienta '{tool_name}' no encontrada"}
            
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        
        return results


# ========================================
# 🚀 APLICACIÓN PRINCIPAL
# ========================================
def main():
    """Inicializa y lanza la aplicación"""
    print("🔄 Cargando perfil profesional...")
    profile = ProfileLoader()
    
    print("💬 Iniciando chat manager...")
    chat_manager = ChatManager(profile)
    
    print("🎨 Lanzando interfaz Gradio...")
    
    # ========================================
    # 🎨 CSS OPTIMIZADO Y CORREGIDO
    # ========================================
    custom_css = """
    /* ==========================================
       CONFIGURACIÓN BASE - Tema Oscuro
       ========================================== */
    .gradio-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        min-height: 100vh;
    }
    
    /* Contenedor principal - permite scroll natural */
    .contain {
        max-width: 1600px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* ==========================================
       ÁREA DE CHAT - Scroll funcional y compacto
       ========================================== */
    #chat-container {
        height: 550px !important;
        max-height: 65vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        padding: 16px !important;
        margin: 0 !important; /* Eliminamos el margin-bottom */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Mensajes del chat */
    #chat-container .message-wrap {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        margin: 10px 0 !important;
        padding: 14px 18px !important;
        max-width: 85% !important;
    }
    
    /* Mensajes del usuario */
    #chat-container .user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        margin-left: auto !important;
        border: none !important;
    }
    
    /* Mensajes del bot */
    #chat-container .bot {
        background: rgba(255, 255, 255, 0.08) !important;
        border-left: 3px solid #667eea !important;
        color: #e8e8e8 !important;
        margin-right: auto !important;
    }
    
    /* ==========================================
       SIDEBAR - Scroll independiente FORZADO
       ========================================== */
    .sidebar-questions {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        height: 600px !important;
        max-height: 70vh !important;
        overflow-y: scroll !important; /* Cambiado de auto a scroll */
        overflow-x: hidden !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        display: block !important;
    }
    
    /* Forzar scroll en el contenedor interno */
    .sidebar-questions > div {
        height: 100% !important;
        overflow-y: auto !important;
    }
    
    /* ==========================================
       SCROLLBARS PERSONALIZADOS
       ========================================== */
    /* Para Webkit (Chrome, Safari, Edge) */
    #chat-container::-webkit-scrollbar,
    .sidebar-questions::-webkit-scrollbar {
        width: 10px;
    }
    
    #chat-container::-webkit-scrollbar-track,
    .sidebar-questions::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        margin: 4px;
    }
    
    #chat-container::-webkit-scrollbar-thumb,
    .sidebar-questions::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.05);
    }
    
    #chat-container::-webkit-scrollbar-thumb:hover,
    .sidebar-questions::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #7c8ef0 0%, #8a5bb0 100%);
    }
    
    /* Para Firefox */
    #chat-container,
    .sidebar-questions {
        scrollbar-width: thin;
        scrollbar-color: #667eea rgba(255, 255, 255, 0.05);
    }
    
    /* ==========================================
       BOTONES DE PREGUNTAS SUGERIDAS
       ========================================== */
    .question-btn {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #e0e0e0 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 13px !important;
        text-align: left !important;
        width: 100% !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.5 !important;
        cursor: pointer !important;
    }
    
    .question-btn:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
        border-color: rgba(102, 126, 234, 0.6) !important;
        transform: translateX(6px) scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    .question-btn:active {
        transform: translateX(6px) scale(0.98) !important;
    }
    
    /* ==========================================
       TIPOGRAFÍA Y TÍTULOS
       ========================================== */
    h1, h2, h3, h4 {
        color: #e8e8e8 !important;
        font-weight: 600 !important;
        margin: 0 0 12px 0 !important;
    }
    
    .header-section {
        padding: 24px 20px !important;
        margin-bottom: 20px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    .header-section h1 {
        font-size: 28px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px !important;
    }
    
    .header-section h3 {
        font-size: 16px !important;
        color: #b0b0b0 !important;
        font-weight: 400 !important;
    }
    
    .section-title {
        color: #667eea !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        margin: 20px 0 12px 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3) !important;
        padding-bottom: 8px !important;
    }
    
    .sidebar-title {
        color: #e8e8e8 !important;
        font-size: 20px !important;
        margin-bottom: 12px !important;
        text-align: center !important;
    }
    
    /* ==========================================
       ÁREA DE INPUT - Alineada y compacta
       ========================================== */
    .input-container {
        padding: 0 !important; /* Eliminamos padding interno */
        background: transparent !important; /* Sin fondo duplicado */
        border-radius: 0 !important;
        margin-top: 12px !important;
        border: none !important; /* Sin borde duplicado */
        display: flex !important;
        gap: 8px !important;
        align-items: stretch !important;
    }
    
    /* Textbox optimizado */
    .input-container textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #e8e8e8 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        font-size: 14px !important;
        resize: none !important;
        min-height: 50px !important;
        max-height: 100px !important;
    }
    
    .input-container textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        outline: none !important;
    }
    
    /* Botón de envío - mismo alto que el textarea */
    .input-container button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0 24px !important;
        font-size: 20px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        min-width: 60px !important;
        height: auto !important;
        align-self: stretch !important;
    }
    
    .input-container button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .input-container button:active {
        transform: scale(0.98) !important;
    }
    
    /* ==========================================
       RESPONSIVE DESIGN
       ========================================== */
    
    /* Tablets (768px - 1024px) */
    @media (max-width: 1024px) {
        #chat-container {
            height: 450px !important;
            max-height: 60vh !important;
        }
        
        .sidebar-questions {
            height: 500px !important;
            max-height: 60vh !important;
        }
        
        .question-btn {
            font-size: 12px !important;
            padding: 10px 14px !important;
        }
        
        .header-section h1 {
            font-size: 24px !important;
        }
        
        .input-container textarea {
            font-size: 13px !important;
            padding: 12px 14px !important;
        }
    }
    
    /* Móviles (hasta 768px) */
    @media (max-width: 768px) {
        .contain {
            padding: 12px;
        }
        
        #chat-container {
            height: 350px !important;
            max-height: 50vh !important;
            margin-bottom: 16px !important;
        }
        
        .sidebar-questions {
            height: 300px !important;
            max-height: 45vh !important;
            margin-top: 16px !important;
        }
        
        .header-section {
            padding: 16px !important;
        }
        
        .header-section h1 {
            font-size: 20px !important;
        }
        
        .header-section h3 {
            font-size: 13px !important;
        }
        
        .question-btn {
            font-size: 11px !important;
            padding: 10px 12px !important;
        }
        
        .section-title {
            font-size: 11px !important;
            margin: 16px 0 10px 0 !important;
        }
        
        .sidebar-title {
            font-size: 16px !important;
        }
        
        .input-container textarea {
            font-size: 13px !important;
            padding: 10px 12px !important;
            min-height: 45px !important;
        }
        
        .input-container button {
            min-width: 50px !important;
            font-size: 18px !important;
        }
    }
    
    /* Móviles pequeños (hasta 480px) */
    @media (max-width: 480px) {
        #chat-container {
            height: 300px !important;
            max-height: 45vh !important;
            padding: 12px !important;
        }
        
        .sidebar-questions {
            height: 250px !important;
            max-height: 40vh !important;
            padding: 16px !important;
        }
        
        .header-section h1 {
            font-size: 18px !important;
        }
        
        .header-section h3 {
            font-size: 12px !important;
        }
        
        #chat-container .message-wrap {
            padding: 10px 14px !important;
            max-width: 90% !important;
        }
        
        .input-container textarea {
            font-size: 12px !important;
            padding: 10px !important;
            min-height: 42px !important;
        }
        
        .input-container button {
            min-width: 45px !important;
            font-size: 16px !important;
            padding: 0 16px !important;
        }
    }
    
    /* ==========================================
       MEJORAS DE ACCESIBILIDAD
       ========================================== */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Mejorar contraste de texto */
    .bot p, .user p {
        line-height: 1.6 !important;
        margin: 0 !important;
    }
    
    /* Focus visible para accesibilidad */
    button:focus-visible,
    textarea:focus-visible {
        outline: 3px solid #667eea !important;
        outline-offset: 2px !important;
    }
    """
    
    # Preguntas sugeridas organizadas por categorías
    preguntas_clave = {
        "🎯 Perfil General": [
            "¿Cuál es tu experiencia profesional?",
            "¿Cuál es tu stack tecnológico principal?",
            "¿Cuántos proyectos has desarrollado?"
        ],
        "💻 Experiencia Backend": [
            "¿Qué proyectos has hecho con Python?",
            "¿Tienes experiencia con FastAPI o Django?",
            "¿Has trabajado con bases de datos?",
            "Muéstrame tu experiencia en APIs"
        ],
        "🎨 Experiencia Frontend": [
            "¿Qué frameworks de frontend dominas?",
            "¿Has trabajado con React o Vue?",
            "Muéstrame proyectos de UI"
        ],
        "🤖 Machine Learning & IA": [
            "¿Tienes experiencia en ML?",
            "¿Qué proyectos de IA has desarrollado?",
            "¿Has trabajado con TensorFlow?",
            "Muéstrame modelos de ML"
        ],
        "🏗️ Arquitectura & DevOps": [
            "¿Experiencia con Docker y Kubernetes?",
            "¿Has trabajado con microservicios?",
            "¿Qué experiencia tienes en DevOps?",
            "Proyectos con arquitectura escalable"
        ],
        "🚀 Proyectos Destacados": [
            "¿Cuáles son tus proyectos más complejos?",
            "¿Has desarrollado apps Full Stack?",
            "¿Proyectos de E-commerce?",
            "Proyectos con procesamiento de datos"
        ]
    }
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        # Header
        with gr.Column(elem_classes="header-section"):
            gr.Markdown(
                f"""
                # 💬 Chat con {profile.name}
                ### Ingeniero de Software | Full Stack Developer | ML Engineer
                """,
                elem_classes="header"
            )
        
        # Contenedor principal con dos columnas
        with gr.Row(equal_height=True):
            # Columna del chat (65%)
            with gr.Column(scale=65):
                chatbot = gr.Chatbot(
                    elem_id="chat-container",
                    type="messages",
                    show_label=False,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"),
                    show_copy_button=True,
                    height=600
                )
                
                with gr.Row(elem_classes="input-container"):
                    with gr.Column(scale=9, min_width=0):
                        msg = gr.Textbox(
                            placeholder="💭 Escribe tu pregunta aquí...",
                            show_label=False,
                            container=False,
                            lines=2,
                            max_lines=4
                        )
                    with gr.Column(scale=1, min_width=60):
                        submit = gr.Button("📤", variant="primary", size="lg")
            
            # Sidebar con preguntas sugeridas (35%)
            with gr.Column(scale=35, elem_classes="sidebar-questions"):
                gr.Markdown("### 📋 Preguntas Sugeridas", elem_classes="sidebar-title")
                gr.Markdown("*💡 Haz clic en cualquier pregunta*", elem_classes="sidebar-subtitle")
                
                # Crear botones por categoría
                for categoria, preguntas in preguntas_clave.items():
                    gr.Markdown(f"**{categoria}**", elem_classes="section-title")
                    for pregunta in preguntas:
                        btn = gr.Button(
                            f"→ {pregunta}",
                            elem_classes="question-btn",
                            size="sm"
                        )
                        # Cuando se hace clic, coloca la pregunta en el textbox
                        btn.click(
                            lambda p=pregunta: p,
                            None,
                            msg,
                            queue=False
                        )
        
        # ========================================
        # LÓGICA DEL CHAT
        # ========================================
        def respond(message, chat_history):
            """Procesa mensaje y genera respuesta"""
            if not message.strip():
                return "", chat_history
            
            # Agregar mensaje del usuario
            chat_history.append({"role": "user", "content": message})
            
            # Obtener respuesta del bot
            bot_message = chat_manager.chat(message, chat_history[:-1])
            
            # Agregar respuesta del bot
            chat_history.append({"role": "assistant", "content": bot_message})
            
            return "", chat_history
        
        # Eventos
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
    
    # Lanzar aplicación
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()