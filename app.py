from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
    PUSHOVER_USER = os.getenv("PUSHOVER_USER")
    MODEL = "gpt-4o-mini"
    
    LINKEDIN_PDF = "me/linkedin.pdf"
    SUMMARY_TXT = "me/summary.txt"
    PROJECTS_CSV = "datasets/resumen/repos_con_tags_dinamicos.csv"
    METADATA_JSON = "datasets/resumen/metadata_dinamica.json"
    
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")


class NotificationService:
    @staticmethod
    def send(message):
        if not Config.PUSHOVER_TOKEN or not Config.PUSHOVER_USER:
            return
        try:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": Config.PUSHOVER_TOKEN,
                    "user": Config.PUSHOVER_USER,
                    "message": message,
                },
                timeout=5
            )
        except Exception as e:
            logger.error(f"Notification error: {e}")


class ProfileLoader:
    def __init__(self, name="Claudio Quispe"):
        self.name = name
        self.linkedin = self._load_linkedin()
        self.summary = self._load_summary()
    
    def _load_linkedin(self):
        try:
            reader = PdfReader(Config.LINKEDIN_PDF)
            return "".join(page.extract_text() or "" for page in reader.pages)
        except FileNotFoundError:
            logger.warning(f"LinkedIn PDF not found: {Config.LINKEDIN_PDF}")
            return "Perfil de LinkedIn no disponible"
    
    def _load_summary(self):
        try:
            with open(Config.SUMMARY_TXT, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Summary not found: {Config.SUMMARY_TXT}")
            return "Resumen profesional no disponible"


class ProjectRepository:
    def __init__(self):
        self.projects_df = self._load_projects()
        self.metadata = self._load_metadata()
    
    def _load_projects(self):
        try:
            return pd.read_csv(Config.PROJECTS_CSV, sep=",")
        except FileNotFoundError:
            logger.warning(f"Projects CSV not found: {Config.PROJECTS_CSV}")
            return pd.DataFrame()
    
    def _load_metadata(self):
        try:
            with open(Config.METADATA_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Metadata not found: {Config.METADATA_JSON}")
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
                        "description": "Categoría: general, backend, frontend, ml, ia",
                        "enum": ["general", "backend", "frontend", "ml", "ia"]
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
            
            tool_function = globals().get(tool_name)
            result = tool_function(**arguments) if tool_function else {"error": f"Tool '{tool_name}' not found"}
            
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        
        return results


def create_gradio_interface(chat_manager: ChatManager, profile: ProfileLoader):
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    #chat-container {
        height: 550px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    .user, .message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
    }
    
    .bot, .message.bot {
        background: rgba(255, 255, 255, 0.12) !important;
        border-left: 3px solid #667eea !important;
        color: #ffffff !important;
    }
    
    .sidebar-questions {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        height: 600px !important;
        overflow-y: auto !important;
    }
    
    .question-btn {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-size: 12px !important;
        text-align: left !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .question-btn:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
        transform: translateX(4px) !important;
    }
    
    .header-section h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
            "¿Tienes experiencia con FastAPI?",
            "Muéstrame tu experiencia en APIs"
        ],
        "🎨 Frontend": [
            "¿Qué frameworks de frontend dominas?",
            "¿Has trabajado con React?",
            "Muéstrame proyectos de UI"
        ],
        "🤖 ML & IA": [
            "¿Tienes experiencia en ML?",
            "¿Qué proyectos de IA has desarrollado?",
            "Muéstrame modelos de ML"
        ],
        "🚀 Destacados": [
            "¿Cuáles son tus proyectos más complejos?",
            "¿Has desarrollado apps Full Stack?",
            "Proyectos con procesamiento de datos"
        ]
    }
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes="header-section"):
            gr.Markdown(f"# 💬 Chat con {profile.name}\n### Desarrollador Web & Data/AI Solutions")
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=65):
                chatbot = gr.Chatbot(
                    elem_id="chat-container",
                    type="messages",
                    show_label=False,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"),
                    height=600
                )
                
                with gr.Row():
                    with gr.Column(scale=9):
                        msg = gr.Textbox(
                            placeholder="💭 Escribe tu pregunta aquí...",
                            show_label=False,
                            lines=2
                        )
                    with gr.Column(scale=1, min_width=60):
                        submit = gr.Button("📤", variant="primary")
            
            with gr.Column(scale=35, elem_classes="sidebar-questions"):
                gr.Markdown("### 📋 Preguntas Sugeridas")
                
                for categoria, preguntas in preguntas_clave.items():
                    gr.Markdown(f"**{categoria}**")
                    for pregunta in preguntas:
                        btn = gr.Button(f"→ {pregunta}", elem_classes="question-btn", size="sm")
                        btn.click(lambda p=pregunta: p, None, msg, queue=False)
        
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


def create_app():
    profile = ProfileLoader()
    chat_manager = ChatManager(profile)
    
    app = FastAPI(
        title="Portfolio Chat API",
        description="Integrated FastAPI + Gradio portfolio chat",
        version="2.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return RedirectResponse(url="/chat")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": "2.0.0"}
    
    @app.post("/api/chat")
    async def chat_endpoint(request: Request):
        try:
            body = await request.json()
            message = body.get("message", "")
            history = body.get("history", [])
            
            if not message.strip():
                return JSONResponse({"error": "Empty message"}, status_code=400)
            
            response = chat_manager.chat(message, history)
            return JSONResponse({"response": response, "status": "success"})
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
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
            logger.error(f"Projects error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @app.get("/api/expertise")
    async def expertise_endpoint(categoria: str = "general"):
        try:
            valid = ["general", "backend", "frontend", "ml", "ia"]
            if categoria not in valid:
                return JSONResponse({"error": f"Invalid category. Use: {valid}"}, status_code=400)
            
            result = get_technical_expertise(categoria)
            return JSONResponse(result)
        except Exception as e:
            logger.error(f"Expertise error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    gradio_app = create_gradio_interface(chat_manager, profile)
    app = gr.mount_gradio_app(app, gradio_app, path="/chat")
    
    return app


def main():
    logger.info("Starting Portfolio Chat Server")
    logger.info(f"Host: {Config.HOST}:{Config.PORT}")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()