from github import Github
from dotenv import load_dotenv
import os
import csv
import base64
import re

# Carga las variables desde tu archivo .env
load_dotenv()

# Obtén el token desde las variables de entorno
token = os.getenv("GITHUB_TOKEN")

# Validación por seguridad
if not token:
    raise SystemExit("⚠️  No se encontró GITHUB_TOKEN en el entorno (.env)")

# Autenticación con GitHub
g = Github(token)

def limpiar_markdown(texto, max_chars=2000):
    """
    Limpia el contenido Markdown y devuelve texto plano optimizado
    Elimina: !, [], {}, "", sintaxis MD, HTML, etc.
    """
    if not texto:
        return ""
    
    # Eliminar bloques de código
    texto = re.sub(r'```[\s\S]*?```', '', texto)
    texto = re.sub(r'`[^`]+`', '', texto)
    
    # Eliminar imágenes ![alt](url) ANTES de procesar enlaces
    texto = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', texto)
    
    # Eliminar enlaces [texto](url) - conservar solo el texto
    texto = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', texto)
    
    # Eliminar corchetes vacíos o sobrantes []
    texto = re.sub(r'\[\s*\]', '', texto)
    texto = re.sub(r'[\[\]]', '', texto)
    
    # Eliminar llaves {} (usadas en templates, variables, etc)
    texto = re.sub(r'\{[^\}]*\}', '', texto)
    texto = re.sub(r'[{}]', '', texto)
    
    # Eliminar comillas dobles excesivas ""
    texto = re.sub(r'"{2,}', '"', texto)
    
    # Eliminar signos de exclamación sobrantes (dejando solo uno si es necesario)
    texto = re.sub(r'!+', '', texto)
    
    # Eliminar encabezados # pero conservar el texto
    texto = re.sub(r'^#{1,6}\s+', '', texto, flags=re.MULTILINE)
    
    # Eliminar énfasis **, __, *, _
    texto = re.sub(r'\*\*([^\*]+)\*\*', r'\1', texto)
    texto = re.sub(r'__([^_]+)__', r'\1', texto)
    texto = re.sub(r'\*([^\*]+)\*', r'\1', texto)
    texto = re.sub(r'_([^_]+)_', r'\1', texto)
    
    # Eliminar asteriscos sobrantes
    texto = re.sub(r'\*+', '', texto)
    
    # Eliminar viñetas de listas - * +
    texto = re.sub(r'^\s*[-*+]\s+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'^\s*\d+\.\s+', '', texto, flags=re.MULTILINE)
    
    # Eliminar líneas horizontales
    texto = re.sub(r'^[-*_]{3,}$', '', texto, flags=re.MULTILINE)
    
    # Eliminar HTML tags y estilos
    texto = re.sub(r'<[^>]+>', '', texto)
    
    # Eliminar blockquotes >
    texto = re.sub(r'^>\s*', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'>', '', texto)
    
    # Eliminar caracteres especiales de markdown sobrantes
    texto = re.sub(r'[~`]', '', texto)
    
    # Limpiar múltiples espacios en blanco
    texto = re.sub(r'\s+', ' ', texto)
    
    # Limpiar múltiples saltos de línea
    texto = re.sub(r'\n{2,}', '\n', texto)
    
    # Eliminar espacios al inicio y final
    texto = texto.strip()
    
    # Limitar a max_chars caracteres
    if len(texto) > max_chars:
        texto = texto[:max_chars].rsplit(' ', 1)[0] + '...'  # Cortar en palabra completa
    
    return texto

# Columnas OPTIMIZADAS para el CSV
columnas = [
    'repo_nombre',
    'es_privado',
    'tiene_readme',
    'documentacion',           # Solo esta columna con texto limpio (max 2000 chars)
    'archivo_origen',
    'fecha_actualizacion',
    'url_repositorio'
]

# Crear el directorio datasets si no existe
os.makedirs('datasets', exist_ok=True)

# Archivo CSV de salida
csv_file = 'datasets/repos_documentacion.csv'

# Abrir CSV para escritura con encoding UTF-8 explícito
with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=columnas)
    writer.writeheader()
    
    # Iterar sobre los repositorios
    contador_exitosos = 0
    contador_sin_readme = 0
    
    # Iterar sobre los repositorios
    for repo in g.get_user().get_repos():
        print(f"📦 Procesando: {repo.full_name}")
        
        try:
            # Intentar obtener el README
            readme = repo.get_readme()
            readme_contenido = base64.b64decode(readme.content).decode('utf-8')
            tiene_readme = True
            
            # Limpiar y optimizar el contenido (máximo 2500 caracteres)
            documentacion_limpia = limpiar_markdown(readme_contenido, max_chars=2500)
            contador_exitosos += 1
            
        except Exception as e:
            print(f"   ⚠️  Sin README")
            documentacion_limpia = "Sin documentación disponible"
            tiene_readme = False
            contador_sin_readme += 1
        
        # Escribir fila en CSV
        writer.writerow({
            'repo_nombre': repo.full_name,
            'es_privado': 'Sí' if repo.private else 'No',
            'tiene_readme': 'Sí' if tiene_readme else 'No',
            'documentacion': documentacion_limpia,
            'archivo_origen': 'README.md' if tiene_readme else 'N/A',
            'fecha_actualizacion': repo.updated_at.strftime('%Y-%m-%d %H:%M:%S') if repo.updated_at else '',
            'url_repositorio': repo.html_url
        })

print(f"\n{'='*60}")
print(f"✅ CSV generado correctamente en: {csv_file}")
