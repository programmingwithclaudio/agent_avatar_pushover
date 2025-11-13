from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from threading import Thread

load_dotenv(override=True)

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
    PUSHOVER_USER = os.getenv("PUSHOVER_USER")
    MODEL = "gpt-4o-mini"
    
    LINKEDIN_PDF = "me/linkedin.pdf"
    SUMMARY_TXT = "me/summary.txt"
    PROJECTS_CSV = "datasets/resumen/repos_con_tags_dinamicos.csv"
    METADATA_JSON = "datasets/resumen/metadata_dinamica.json"
    
    GRADIO_PORT = 7860
    FASTAPI_PORT = 8000


class NotificationService:
    @staticmethod
    def send(message):
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


class ProfileLoader:
    def __init__(self, name="Claudio Quispe"):
        self.name = name
        self.linkedin = self._load_linkedin()
        self.summary = self._load_summary()
    
    def _load_linkedin(self):
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
        try:
            with open(Config.SUMMARY_TXT, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ No se encontró {Config.SUMMARY_TXT}")
            return "Resumen profesional no disponible"


class ProjectRepository:
    def __init__(self):
        self.projects_df = self._load_projects()
        self.metadata = self._load_metadata()
    
    def _load_projects(self):
        try:
            return pd.read_csv(Config.PROJECTS_CSV, sep=",")
        except FileNotFoundError:
            print(f"⚠️ No se encontró {Config.PROJECTS_CSV}")
            return pd.DataFrame()
    
    def _load_metadata(self):
        try:
            with open(Config.METADATA_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ No se encontró {Config.METADATA_JSON}")
            return {}
    
    def search(self, dominio=None, tecnologia=None, tipo_proyecto=None, incluye_ml=False, limit=5):
        if self.projects_df.empty:
            return {"error": "No hay proyectos disponibles"}
        
        proyectos_encontrados = []
        
        for idx, row in self.projects_df.iterrows():
            try:
                clasificacion = json.loads(row['clasificacion_dinamica'])
                
                if not self._match_filters(clasificacion, dominio, tecnologia, tipo_proyecto, incluye_ml):
                    continue
                
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
        if dominio and clasificacion.get('dominio_aplicacion', '').lower() != dominio.lower():
            return False
        
        if tecnologia:
            todas_tech = (
                clasificacion.get('tecnologias_backend', []) +
                clasificacion.get('tecnologias_frontend', []) +
                clasificacion.get('bases_datos', []) +
                clasificacion.get('devops_cloud', [])
            )
            if not any(tecnologia.lower() in tech.lower() for tech in todas_tech):
                return False
        
        if tipo_proyecto:
            if not any(tipo_proyecto.lower() in tipo.lower() for tipo in clasificacion.get('tipo_proyecto', [])):
                return False
        
        if incluye_ml and not clasificacion.get('ml_ia', []):
            return False
        
        return True
    
    def get_expertise(self, categoria="general"):
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


def record_user_details(email, name="Nombre no indicado", notes="no proporcionadas"):
    NotificationService.send(f"📧 Contacto: {name} | Email: {email} | Notas: {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    NotificationService.send(f"❓ Pregunta sin respuesta: {question}")
    return {"recorded": "ok"}

def search_projects(dominio=None, tecnologia=None, tipo_proyecto=None, incluye_ml=False, limit=5):
    repo = ProjectRepository()
    return repo.search(dominio, tecnologia, tipo_proyecto, incluye_ml, limit)

def get_technical_expertise(categoria="general"):
    repo = ProjectRepository()
    return repo.get_expertise(categoria)


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
            "description": "Busca proyectos específicos por dominio, tecnología o tipo.",
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
            "description": "Muestra expertise técnico y estadísticas del portfolio.",
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


class ChatManager:
    def __init__(self, profile: ProfileLoader):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.profile = profile
    
    def build_system_prompt(self):
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
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ] + history + [
            {"role": "user", "content": message}
        ]
        
        while True:
            response = self.client.chat.completions.create(
                model=Config.MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA
            )
            
            choice = response.choices[0]
            
            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                tool_results = self._execute_tools(choice.message.tool_calls)
                messages.extend(tool_results)
            else:
                return choice.message.content
    
    def _execute_tools(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"🔧 Ejecutando: {tool_name} con {arguments}", flush=True)
            
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


def create_fastapi_app(chat_manager: ChatManager):
    app = FastAPI(title="Portfolio Chat API", version="1.0")
    
    @app.get("/")
    async def root():
        return {
            "message": "Portfolio Chat API",
            "endpoints": {
                "chat": "/api/chat",
                "projects": "/api/projects",
                "expertise": "/api/expertise"
            }
        }
    
    @app.post("/api/chat")
    async def chat_endpoint(request: dict):
        try:
            message = request.get("message", "")
            history = request.get("history", [])
            
            response = chat_manager.chat(message, history)
            
            return JSONResponse({
                "response": response,
                "status": "success"
            })
        except Exception as e:
            return JSONResponse({
                "error": str(e),
                "status": "error"
            }, status_code=500)
    
    @app.get("/api/projects")
    async def projects_endpoint(
        dominio: str = None,
        tecnologia: str = None,
        tipo_proyecto: str = None,
        incluye_ml: bool = False,
        limit: int = 5
    ):
        try:
            result = search_projects(dominio, tecnologia, tipo_proyecto, incluye_ml, limit)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({
                "error": str(e),
                "status": "error"
            }, status_code=500)
    
    @app.get("/api/expertise")
    async def expertise_endpoint(categoria: str = "general"):
        try:
            result = get_technical_expertise(categoria)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({
                "error": str(e),
                "status": "error"
            }, status_code=500)
    
    return app


def create_gradio_app(chat_manager: ChatManager, profile: ProfileLoader):
    
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        min-height: 100vh;
    }
    
    .contain {
        max-width: 1600px;
        margin: 0 auto;
        padding: 20px;
    }
    
    #chat-container {
        height: 550px !important;
        max-height: 65vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        padding: 16px !important;
        margin: 0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    
    #chat-container .message-wrap {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        margin: 10px 0 !important;
        padding: 14px 18px !important;
        max-width: 85% !important;
    }
    
    #chat-container .user,
    #chat-container .message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        border: none !important;
    }
    
    #chat-container .user p,
    #chat-container .user span,
    #chat-container .user div,
    #chat-container .message.user p,
    #chat-container .message.user span,
    #chat-container .message.user div {
        color: #ffffff !important;
    }
    
    #chat-container .bot,
    #chat-container .message.bot {
        background: rgba(255, 255, 255, 0.12) !important;
        border-left: 3px solid #667eea !important;
        color: #ffffff !important;
        margin-right: auto !important;
        margin-left: 0 !important;
    }
    
    #chat-container .bot p,
    #chat-container .bot span,
    #chat-container .bot div,
    #chat-container .message.bot p,
    #chat-container .message.bot span,
    #chat-container .message.bot div {
        color: #ffffff !important;
    }
    
    .sidebar-questions {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 16px !important;
        padding: 0 !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        height: 600px !important;
        max-height: 70vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        display: block !important;
    }
    .category-title {
        color: #667eea !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        margin: 16px 0 8px 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        border-bottom: 1px solid rgba(102, 126, 234, 0.3) !important;
        padding-bottom: 4px !important;
        line-height: 1 !important;
    }
    .sidebar-questions .category-title,
    .sidebar-questions .category-title * {
        color: #667eea !important;
    }
    
    .category-section {
        margin-bottom: 16px !important;
        padding: 0 12px !important;
    }

    .questions-container {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        margin-bottom: 8px !important;
    } 
    
    .sidebar-questions .markdown,
    .sidebar-questions .prose {
        margin: 4px 0 !important;
        padding: 0 !important;
        /* Añadir para evitar que herede colores incorrectos */
        color: inherit !important;
    }
    .category-title .markdown,
    .category-title .prose {
        color: #667eea !important;
    }
    .sidebar-questions .markdown p,
    .sidebar-questions .prose p {
        color: inherit !important;
    }
    #chat-container::-webkit-scrollbar,
    .sidebar-questions::-webkit-scrollbar {
        width: 10px;
    }
    
    #chat-container::-webkit-scrollbar-track,
    .sidebar-questions::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        margin: 8px;
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
    
    #chat-container,
    .sidebar-questions {
        scrollbar-width: thin;
        scrollbar-color: #667eea rgba(255, 255, 255, 0.05);
    }
    
    .question-btn {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 0 !important; /* Eliminamos márgenes verticales */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 12px !important;
        text-align: left !important;
        width: 100% !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.4 !important;
        cursor: pointer !important;
        flex-shrink: 0 !important;
    }
    
    .question-btn:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
        border-color: rgba(102, 126, 234, 0.6) !important;
        transform: translateX(4px) scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        color: #ffffff !important;
    }
    
    .question-btn:active {
        transform: translateX(6px) scale(0.98) !important;
    }
    
    h1, h2, h3, h4 {
        color: #e8e8e8 !important;
        font-weight: 600 !important;
        margin: 0 0 6px 0 !important;
    }
    
    .header-section {
        padding: 18px !important;
        margin-bottom: 14px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    .header-section h1 {
        font-size: 26px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px !important;
    }
    
    .header-section h3 {
        font-size: 15px !important;
        color: #158f56 !important;
        font-weight: 400 !important;
    }
    
    .section-title {
        color: #667eea !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        margin: 12px 0 6px 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        border-bottom: 1px solid rgba(102, 126, 234, 0.3) !important;
        padding-bottom: 2px !important;
        line-height: 1 !important;
    }
    
    .section-title:first-of-type {
        margin-top: 0 !important;
    }
    
    .sidebar-title {
        color: #e8e8e8 !important;
        font-size: 17px !important;
        margin: 12px 0 6px 0 !important;
        text-align: center !important;
    }
    
    .sidebar-subtitle {
        font-size: 10px !important;
        color: #888 !important;
        text-align: center !important;
        margin: 0 0 12px 0 !important;
        font-style: italic !important;
    }
    
    .input-container {
        padding: 0 !important;
        background: #1f2937;
        border-radius: 0 !important;
        margin-top: 12px !important;
        border: none !important;
        display: flex !important;
        gap: 8px !important;
        align-items: stretch !important;
    }
    
    .input-container textarea {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        font-size: 14px !important;
        resize: none !important;
        min-height: 50px !important;
        max-height: 100px !important;
    }
    
    .input-container textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .input-container textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        outline: none !important;
        background: rgba(255, 255, 255, 0.12) !important;
    }
    
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
    
    @media (max-width: 768px) {
        .contain {
            padding: 12px;
        }
        
        #chat-container {
            height: 350px !important;
            max-height: 50vh !important;
        }
        
        .sidebar-questions {
            height: 300px !important;
            max-height: 45vh !important;
        }
        
        .sidebar-questions > * {
            padding: 0 10px !important;
        }
        
        .question-btn {
            font-size: 11px !important;
            padding: 7px 10px !important;
        }
        
        .section-title {
            font-size: 9px !important;
            margin: 10px 0 4px 0 !important;
        }
    }
    
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .bot p, .user p,
    .message.bot p, .message.user p {
        line-height: 1.6 !important;
        margin: 0 !important;
        color: inherit !important;
    }
    
    #chat-container * {
        color: inherit !important;
    }
    
    button:focus-visible,
    textarea:focus-visible {
        outline: 3px solid #667eea !important;
        outline-offset: 2px !important;
    }
    """
    
    preguntas_clave = {
        "🎯 Perfil General": [
            "¿Cuál es tu experiencia profesional?",
            "¿Cuál es tu stack tecnológico principal?",
            "¿Cuántos proyectos has desarrollado?"
        ],
        "💻 Backend": [
            "¿Qué proyectos has hecho con Python?",
            "¿Tienes experiencia con FastAPI o Django?",
            "¿Has trabajado con bases de datos?",
            "Muéstrame tu experiencia en APIs"
        ],
        "🎨 Frontend": [
            "¿Qué frameworks de frontend dominas?",
            "¿Has trabajado con React o Vue?",
            "Muéstrame proyectos de UI"
        ],
        "🤖 ML & IA": [
            "¿Tienes experiencia en ML?",
            "¿Qué proyectos de IA has desarrollado?",
            "¿Has trabajado con TensorFlow?",
            "Muéstrame modelos de ML"
        ],
        "🏗️ Arquitectura": [
            "¿Experiencia con Docker y Kubernetes?",
            "¿Has trabajado con microservicios?",
            "¿Qué experiencia tienes en DevOps?",
            "Proyectos con arquitectura escalable"
        ],
        "🚀 Destacados": [
            "¿Cuáles son tus proyectos más complejos?",
            "¿Has desarrollado apps Full Stack?",
            "¿Proyectos de E-commerce?",
            "Proyectos con procesamiento de datos"
        ]
    }
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes="header-section"):
            gr.Markdown(
                f"""
                # 💬 Chat con {profile.name}
                ### Desarrollador Web & Data/AI Solutions | Python · SQL/NoSQL
                """,
                elem_classes="header"
            )
        
        with gr.Row(equal_height=True):
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
            
            with gr.Column(scale=35, elem_classes="sidebar-questions"):
                gr.Markdown("### 📋 Preguntas Sugeridas", elem_classes="sidebar-title")
                gr.Markdown("*💡 Haz clic en cualquier pregunta*", elem_classes="sidebar-subtitle")
                
                for categoria, preguntas in preguntas_clave.items():
                    gr.Markdown(f"**{categoria}**", elem_classes="category-title")
                    for pregunta in preguntas:
                        btn = gr.Button(
                            f"→ {pregunta}",
                            elem_classes="question-btn",
                            size="sm"
                        )
                        btn.click(
                            lambda p=pregunta: p,
                            None,
                            msg,
                            queue=False
                        )
        
        def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            
            chat_history.append({"role": "user", "content": message})
            bot_message = chat_manager.chat(message, chat_history[:-1])
            chat_history.append({"role": "assistant", "content": bot_message})
            
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
    
    return demo


def run_fastapi_server(app):
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=Config.FASTAPI_PORT,
        log_level="info"
    )


def main():
    print("=" * 60)
    print("🚀 INICIANDO PORTFOLIO CHAT - DUAL MODE")
    print("=" * 60)
    
    print("\n🔄 Cargando perfil profesional...")
    profile = ProfileLoader()
    
    print("💬 Iniciando chat manager...")
    chat_manager = ChatManager(profile)
    
    print("\n📡 Configurando servidores...")
    
    fastapi_app = create_fastapi_app(chat_manager)
    
    print(f"🌐 Iniciando FastAPI en puerto {Config.FASTAPI_PORT}...")
    fastapi_thread = Thread(target=run_fastapi_server, args=(fastapi_app,), daemon=True)
    fastapi_thread.start()
    
    print(f"🎨 Iniciando Gradio en puerto {Config.GRADIO_PORT}...")
    gradio_app = create_gradio_app(chat_manager, profile)
    
    print("\n" + "=" * 60)
    print("✅ SERVIDORES ACTIVOS")
    print("=" * 60)
    print(f"📊 Gradio UI:    http://127.0.0.1:{Config.GRADIO_PORT}")
    print(f"🔌 FastAPI:      http://127.0.0.1:{Config.FASTAPI_PORT}")
    print(f"📖 API Docs:     http://127.0.0.1:{Config.FASTAPI_PORT}/docs")
    print("=" * 60)
    print("\n💡 ENDPOINTS DISPONIBLES:")
    print(f"   POST http://127.0.0.1:{Config.FASTAPI_PORT}/api/chat")
    print(f"   GET  http://127.0.0.1:{Config.FASTAPI_PORT}/api/projects")
    print(f"   GET  http://127.0.0.1:{Config.FASTAPI_PORT}/api/expertise")
    print("=" * 60 + "\n")
    
    gradio_app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=Config.GRADIO_PORT,
        show_error=True
    )


if __name__ == "__main__":
    main()
        