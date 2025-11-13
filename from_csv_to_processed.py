import pandas as pd
import json
from collections import Counter, defaultdict
import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import time

# ========================================
# CONFIGURACIÓN
# ========================================
load_dotenv()

archivo_repos_entrada = "datasets/repos_documentacion.csv"
archivo_repos_con_tags = "datasets/resumen/repos_con_tags_dinamicos.csv"
archivo_metadata_salida = "datasets/resumen/metadata_dinamica.json"

key_ia = os.getenv("DEEPSEEK_API_KEY")

# ✅ CAMBIO CRÍTICO: Configurar el cliente para DeepSeek
client = OpenAI(
    api_key=key_ia,
    base_url="https://api.deepseek.com"  # ← Esto es lo que faltaba
)

# ========================================
# FUNCTION CALLING: Definir herramientas
# ========================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "clasificar_proyecto",
            "description": "Clasifica un proyecto de software identificando sus tecnologías, propósito, dominio y características principales",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposito_principal": {
                        "type": "string",
                        "description": "El objetivo principal del proyecto en una frase clara (ej: 'API para gestión de inventario', 'Dashboard de análisis financiero')"
                    },
                    "dominio_aplicacion": {
                        "type": "string",
                        "description": "El dominio o industria (ej: 'E-commerce', 'Finanzas', 'Salud', 'Educación', 'DevOps', 'Data Science')"
                    },
                    "tipo_proyecto": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tipo(s) de proyecto (ej: 'API REST', 'Full Stack Web', 'CLI Tool', 'Bot', 'Librería', 'Dashboard', 'Microservicio', 'Scraper', 'Pipeline de datos')"
                    },
                    "tecnologias_backend": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Frameworks y tecnologías backend identificadas (ej: 'FastAPI', 'Django', 'Express.js', 'Spring Boot', 'Node.js')"
                    },
                    "tecnologias_frontend": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Frameworks y tecnologías frontend identificadas (ej: 'React', 'Vue.js', 'Angular', 'Next.js', 'Tailwind CSS')"
                    },
                    "bases_datos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bases de datos utilizadas (ej: 'PostgreSQL', 'MongoDB', 'Redis', 'MySQL', 'Supabase')"
                    },
                    "ml_ia": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tecnologías de ML/IA si aplica (ej: 'TensorFlow', 'PyTorch', 'OpenAI API', 'LangChain', 'Scikit-learn', 'Hugging Face', 'NLP', 'Computer Vision')"
                    },
                    "devops_cloud": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Herramientas DevOps y cloud (ej: 'Docker', 'Kubernetes', 'AWS', 'GitHub Actions', 'Terraform', 'CI/CD')"
                    },
                    "funcionalidades_clave": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Funcionalidades principales (ej: 'Autenticación JWT', 'Procesamiento de pagos', 'Chat en tiempo real', 'Generación de reportes', 'Notificaciones push')"
                    },
                    "lenguajes_programacion": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lenguajes de programación principales (ej: 'Python', 'JavaScript', 'TypeScript', 'Go', 'Rust', 'Java')"
                    },
                    "tags_adicionales": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Otros tags relevantes que no encajan en categorías anteriores (ej: 'Open Source', 'Producción', 'Experimental', 'Template', 'Monorepo')"
                    }
                },
                "required": ["proposito_principal", "dominio_aplicacion", "tipo_proyecto"]
            }
        }
    }
]

# ========================================
# 1. CLASIFICADOR DINÁMICO CON FUNCTION CALLING
# ========================================

def clasificar_proyecto_dinamico(nombre, descripcion, documentacion, max_reintentos=3):
    """
    Usa function calling para que la IA clasifique dinámicamente el proyecto
    """
    # Limitar documentación
    doc_resumida = str(documentacion)[:4000] if pd.notna(documentacion) else ""
    
    prompt = f"""Analiza este proyecto de GitHub a profundidad y clasifícalo usando la función 'clasificar_proyecto'.

**INFORMACIÓN DEL PROYECTO:**
- Nombre del repositorio: {nombre}
- Descripción breve: {descripcion}
- Documentación completa:
{doc_resumida}

**INSTRUCCIONES:**
1. Lee TODO el contenido cuidadosamente
2. Identifica el propósito REAL del proyecto (no asumas, lee la documentación)
3. Extrae TODAS las tecnologías mencionadas, no inventes ninguna
4. Si no hay información sobre alguna categoría, deja el array vacío []
5. Sé específico: no digas solo "backend", di el framework exacto
6. Para ML/IA: identifica si hay modelos, librerías de ML, APIs de IA
7. Para funcionalidades: identifica características técnicas concretas

**IMPORTANTE:** Usa la función 'clasificar_proyecto' para devolver la clasificación estructurada."""

    for intento in range(max_reintentos):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un arquitecto de software experto que analiza proyectos técnicamente. Identificas con precisión tecnologías, frameworks, propósito y dominio de aplicación. NO asumes, solo reportas lo que está documentado."
                    },
                    {"role": "user", "content": prompt}
                ],
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "clasificar_proyecto"}},
                temperature=0.2
            )
            
            # Extraer el function call
            message = response.choices[0].message
            
            if message.tool_calls:
                function_call = message.tool_calls[0]
                argumentos = json.loads(function_call.function.arguments)
                return argumentos
            else:
                # Fallback si no usa function calling
                if intento < max_reintentos - 1:
                    time.sleep(2)
                    continue
                else:
                    return crear_clasificacion_vacia()
                
        except Exception as e:
            print(f"⚠️ Error en intento {intento + 1}: {e}")
            if intento < max_reintentos - 1:
                time.sleep(2)
            else:
                return crear_clasificacion_vacia()

def crear_clasificacion_vacia():
    """Clasificación por defecto en caso de error"""
    return {
        "proposito_principal": "Sin información suficiente",
        "dominio_aplicacion": "No clasificado",
        "tipo_proyecto": [],
        "tecnologias_backend": [],
        "tecnologias_frontend": [],
        "bases_datos": [],
        "ml_ia": [],
        "devops_cloud": [],
        "funcionalidades_clave": [],
        "lenguajes_programacion": [],
        "tags_adicionales": []
    }

# ========================================
# 2. PROCESADOR DE PROYECTOS
# ========================================

def procesar_todos_los_proyectos():
    """
    Lee el CSV y clasifica cada proyecto dinámicamente con guardado progresivo
    """
    print("🤖 Iniciando clasificación DINÁMICA con Function Calling...\n")
    
    if not os.path.exists(archivo_repos_entrada):
        raise FileNotFoundError(f"No se encontró: {archivo_repos_entrada}")
    
    data = pd.read_csv(archivo_repos_entrada, sep=",")
    
    # Verificar si ya existe archivo con progreso previo
    os.makedirs(os.path.dirname(archivo_repos_con_tags), exist_ok=True)
    
    if os.path.exists(archivo_repos_con_tags):
        print("📂 Encontrado progreso previo, cargando...")
        data_previa = pd.read_csv(archivo_repos_con_tags, sep=",")
        if 'clasificacion_dinamica' in data_previa.columns:
            data = data_previa
            print("✅ Progreso cargado. Continuando desde donde quedó.\n")
    else:
        # Agregar columna para clasificación JSON
        data['clasificacion_dinamica'] = ""
    
    total = len(data)
    procesados = 0
    
    for idx, row in data.iterrows():
        # Saltar si ya está procesado
        if pd.notna(data.at[idx, 'clasificacion_dinamica']) and data.at[idx, 'clasificacion_dinamica'] != "":
            try:
                # Verificar que sea JSON válido
                json.loads(data.at[idx, 'clasificacion_dinamica'])
                procesados += 1
                continue
            except:
                pass  # Si no es JSON válido, reprocesar
        
        nombre = row.get('url_repositorio', '').split('/')[-1] if pd.notna(row.get('url_repositorio')) else "Sin nombre"
        descripcion = str(row.get('documentacion_resumen', '')) if 'documentacion_resumen' in data.columns else ""
        documentacion = str(row.get('documentacion', ''))
        
        print(f"🔍 [{idx + 1}/{total}] Analizando: {nombre[:60]}...")
        
        clasificacion = clasificar_proyecto_dinamico(nombre, descripcion, documentacion)
        data.at[idx, 'clasificacion_dinamica'] = json.dumps(clasificacion, ensure_ascii=False)
        
        # Mostrar resumen
        print(f"   📌 Propósito: {clasificacion.get('proposito_principal', 'N/A')[:70]}")
        print(f"   🏢 Dominio: {clasificacion.get('dominio_aplicacion', 'N/A')}")
        print(f"   🔧 Tipo: {', '.join(clasificacion.get('tipo_proyecto', []))}")
        
        # Mostrar tecnologías encontradas
        tech_count = (
            len(clasificacion.get('tecnologias_backend', [])) +
            len(clasificacion.get('tecnologias_frontend', [])) +
            len(clasificacion.get('bases_datos', [])) +
            len(clasificacion.get('ml_ia', []))
        )
        print(f"   ✨ {tech_count} tecnologías identificadas")
        
        procesados += 1
        
        # 💾 GUARDAR CADA 5 PROYECTOS (ajustable)
        if procesados % 5 == 0:
            data.to_csv(archivo_repos_con_tags, index=False, encoding='utf-8-sig')
            print(f"   💾 Progreso guardado ({procesados}/{total})\n")
        else:
            print()
        
        # Pausa
        time.sleep(0.7)
    
    # Guardar final
    data.to_csv(archivo_repos_con_tags, index=False, encoding='utf-8-sig')
    
    print(f"✅ Clasificación completa. Guardado en: {archivo_repos_con_tags}\n")
    
    return data

# ========================================
# 3. GENERADOR DE METADATA INTELIGENTE
# ========================================

def generar_metadata_dinamica(data):
    """
    Genera metadata analizando las clasificaciones dinámicas
    """
    print("📊 Generando metadata con análisis inteligente...\n")
    
    # Contadores por categoría
    contadores = {
        'dominios': Counter(),
        'tipos_proyecto': Counter(),
        'backend': Counter(),
        'frontend': Counter(),
        'bases_datos': Counter(),
        'ml_ia': Counter(),
        'devops': Counter(),
        'funcionalidades': Counter(),
        'lenguajes': Counter(),
        'tags_adicionales': Counter()
    }
    
    propositos = []
    
    for clasificacion_json in data['clasificacion_dinamica']:
        try:
            clasif = json.loads(clasificacion_json)
            
            # Recopilar datos
            contadores['dominios'][clasif.get('dominio_aplicacion', 'No clasificado')] += 1
            propositos.append(clasif.get('proposito_principal', ''))
            
            for tipo in clasif.get('tipo_proyecto', []):
                contadores['tipos_proyecto'][tipo] += 1
            
            for tech in clasif.get('tecnologias_backend', []):
                contadores['backend'][tech] += 1
                
            for tech in clasif.get('tecnologias_frontend', []):
                contadores['frontend'][tech] += 1
                
            for db in clasif.get('bases_datos', []):
                contadores['bases_datos'][db] += 1
                
            for ml in clasif.get('ml_ia', []):
                contadores['ml_ia'][ml] += 1
                
            for devops in clasif.get('devops_cloud', []):
                contadores['devops'][devops] += 1
                
            for func in clasif.get('funcionalidades_clave', []):
                contadores['funcionalidades'][func] += 1
                
            for lang in clasif.get('lenguajes_programacion', []):
                contadores['lenguajes'][lang] += 1
                
            for tag in clasif.get('tags_adicionales', []):
                contadores['tags_adicionales'][tag] += 1
                
        except Exception as e:
            continue
    
    # Construir metadata
    metadata = {
        'total_proyectos': len(data),
        'fecha_generacion': pd.Timestamp.now().isoformat(),
        'dominios_aplicacion': dict(contadores['dominios'].most_common()),
        'tipos_proyecto': dict(contadores['tipos_proyecto'].most_common()),
        'top_tecnologias': {
            'backend': dict(contadores['backend'].most_common(15)),
            'frontend': dict(contadores['frontend'].most_common(15)),
            'bases_datos': dict(contadores['bases_datos'].most_common(10)),
            'ml_ia': dict(contadores['ml_ia'].most_common(15)),
            'devops_cloud': dict(contadores['devops'].most_common(15))
        },
        'funcionalidades_mas_comunes': dict(contadores['funcionalidades'].most_common(20)),
        'lenguajes_programacion': dict(contadores['lenguajes'].most_common(10)),
        'estadisticas': {
            'proyectos_con_backend': sum(1 for c in data['clasificacion_dinamica'] if len(json.loads(c).get('tecnologias_backend', [])) > 0),
            'proyectos_con_frontend': sum(1 for c in data['clasificacion_dinamica'] if len(json.loads(c).get('tecnologias_frontend', [])) > 0),
            'proyectos_con_ml_ia': sum(1 for c in data['clasificacion_dinamica'] if len(json.loads(c).get('ml_ia', [])) > 0),
            'proyectos_full_stack': sum(1 for c in data['clasificacion_dinamica'] 
                if len(json.loads(c).get('tecnologias_backend', [])) > 0 and 
                   len(json.loads(c).get('tecnologias_frontend', [])) > 0)
        }
    }
    
    # Guardar
    os.makedirs(os.path.dirname(archivo_metadata_salida), exist_ok=True)
    with open(archivo_metadata_salida, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metadata generada en: {archivo_metadata_salida}\n")
    
    # Mostrar resumen visual
    mostrar_resumen_visual(metadata)
    
    return metadata

def mostrar_resumen_visual(metadata):
    """Muestra un resumen visual bonito"""
    print("=" * 80)
    print(f"📊 ANÁLISIS INTELIGENTE DE {metadata['total_proyectos']} PROYECTOS")
    print("=" * 80)
    
    # Estadísticas generales
    stats = metadata['estadisticas']
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print("-" * 80)
    print(f"   Backend:     {stats['proyectos_con_backend']} proyectos ({stats['proyectos_con_backend']/metadata['total_proyectos']*100:.1f}%)")
    print(f"   Frontend:    {stats['proyectos_con_frontend']} proyectos ({stats['proyectos_con_frontend']/metadata['total_proyectos']*100:.1f}%)")
    print(f"   Full Stack:  {stats['proyectos_full_stack']} proyectos ({stats['proyectos_full_stack']/metadata['total_proyectos']*100:.1f}%)")
    print(f"   ML/IA:       {stats['proyectos_con_ml_ia']} proyectos ({stats['proyectos_con_ml_ia']/metadata['total_proyectos']*100:.1f}%)")
    
    # Dominios
    print(f"\n🏢 TOP DOMINIOS DE APLICACIÓN:")
    print("-" * 80)
    for dominio, count in list(metadata['dominios_aplicacion'].items())[:10]:
        porcentaje = (count / metadata['total_proyectos']) * 100
        barra = "█" * int(porcentaje / 2)
        print(f"{dominio:30} | {count:3} proyectos ({porcentaje:5.1f}%) {barra}")
    
    # Backend
    if metadata['top_tecnologias']['backend']:
        print(f"\n⚙️  TOP TECNOLOGÍAS BACKEND:")
        print("-" * 80)
        for tech, count in list(metadata['top_tecnologias']['backend'].items())[:10]:
            porcentaje = (count / metadata['total_proyectos']) * 100
            barra = "█" * int(porcentaje / 2)
            print(f"{tech:30} | {count:3} proyectos ({porcentaje:5.1f}%) {barra}")
    
    # ML/IA
    if metadata['top_tecnologias']['ml_ia']:
        print(f"\n🤖 TECNOLOGÍAS ML/IA:")
        print("-" * 80)
        for tech, count in list(metadata['top_tecnologias']['ml_ia'].items())[:10]:
            porcentaje = (count / metadata['total_proyectos']) * 100
            barra = "█" * int(porcentaje / 2)
            print(f"{tech:30} | {count:3} proyectos ({porcentaje:5.1f}%) {barra}")
    
    print("=" * 80)

# ========================================
# 4. EJECUTAR
# ========================================

if __name__ == "__main__":
    try:
        # Clasificar con IA dinámica
        data_clasificada = procesar_todos_los_proyectos()
        
        # Generar metadata (siempre al final para tener datos completos)
        metadata = generar_metadata_dinamica(data_clasificada)
        
        print("\n🎉 ¡Proceso completado exitosamente!")
        print(f"\n📁 Archivos generados:")
        print(f"   - {archivo_repos_con_tags}")
        print(f"   - {archivo_metadata_salida}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario.")
        print("💾 El progreso ha sido guardado automáticamente.")
        print("   Puedes reanudar ejecutando el script nuevamente.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💾 Revisa el archivo de salida - el progreso parcial fue guardado.\n")
        import traceback
        traceback.print_exc()