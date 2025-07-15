# Chat RAG - Sistema de Consulta DOF

Sistema de chat inteligente que permite consultar documentos del **Diario Oficial de la Federación (DOF)** mediante tecnología **RAG (Retrieval-Augmented Generation)**. Utiliza búsqueda semántica avanzada con reranking temporal para encontrar información relevante y genera respuestas contextuales respaldadas por fuentes específicas.

## ¿Qué hace?

**Conversación inteligente con documentos oficiales**: Haz preguntas en lenguaje natural y obtén respuestas precisas extraídas directamente de documentos del DOF, con referencias a las fuentes utilizadas y análisis de actualidad.

**Ejemplo de uso:**
- Pregunta: *"¿Qué decretos relacionados con infraestructura se publicaron en enero 2025?"*
- Respuesta: Información específica con enlaces a documentos exactos del DOF, indicadores de actualidad y contexto temporal

## ✨ Características Principales

- 🤖 **Chat conversacional** con historial embebido y gestión de fuentes integrada
- 🔍 **Búsqueda semántica avanzada** con reranking temporal basado en actualidad
- 📚 **Fuentes embebidas** con fragmentos expandibles y navegación fluida
- ⚙️ **Configuración dinámica** del número de fuentes consultadas (1-10)
- 📅 **Análisis temporal** con indicadores de actualidad y relevancia
- 🎨 **Interfaz moderna** con scroll automático y diseño responsivo optimizado

## 🏗️ Arquitectura Técnica

```
┌─ Interface (Gradio 5.34.2) ─────────────────┐
│ Chat UI + Fuentes Embebidas + Scroll         │
├─ RAG Pipeline ──────────────────────────────┤
│ 1. Query → Embedding (ModernBERT-base)      │
│ 2. Vector Search (DuckDB + VSS)             │
│ 3. Context Assembly                         │
│ 4. LLM Generation (Gemini/OpenAI/Claude)    │
├─ Data Layer ────────────────────────────────┤
│ DuckDB + VSS Extension                      │
│ Embeddings: 768-dim vectors                 │
│ Documents: DOF metadata + text chunks       │
└─────────────────────────────────────────────┘
```

### Stack Tecnológico

- **Frontend**: Gradio 5.34.2 con ChatInterface y CSS customizado
- **Vector DB**: DuckDB con extensión VSS (Vector Similarity Search)
- **Embeddings**: nomic-ai/modernbert-embed-base (768 dimensiones)
- **LLM**: Gemini 2.0 Flash (por defecto), OpenAI GPT-4o-mini, Claude 3.5 Sonnet, Ollama
- **Processing**: Python 3.12+ con arquitectura modular optimizada
- **Logging**: Logging estructurado con archivos rotativos

## 🚀 Inicio Rápido

### 1. Prerrequisitos
- Python 3.12+
- Base de datos DOF creada con embeddings (desde proyecto principal)
- API key de al menos un proveedor LLM

### 2. Instalación
```bash
# Instalar dependencias con uv
uv add openai
uv add python-dotenv
uv add duckdb
uv add "gradio<=5.34.2"
uv add sentence-transformers
```

**Nota importante**: El proyecto usa Gradio 5.34.2 específicamente. Versiones superiores pueden tener problemas de compatibilidad con la interfaz de chat.

### 3. Configuración
```bash
# Desde la raíz del proyecto dof-rag
cd chatbot_gradio

# Crear .env en la raíz del proyecto
echo "GEMINI_API_KEY=tu_api_key_aqui" >> ../.env
echo "OPENAI_API_KEY=tu_openai_key" >> ../.env
echo "ANTHROPIC_API_KEY=tu_anthropic_key" >> ../.env
```

### 4. Ejecutar
```bash
# Desde el directorio chatbot_gradio
python app.py

# O con UV
uv run python app.py

# URL: http://localhost:8888
```

## ⚙️ Configuración de LLM

### Proveedores Soportados
| Proveedor | Variable | Modelo | Límite/min |
|-----------|----------|--------|------------|
| **Gemini** (default) | `GEMINI_API_KEY` | gemini-2.0-flash | 15 |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | 10 |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-5-sonnet | 5 |
| Ollama | (local) | llama3.1 | ∞ |

Configura en `.env` en la **raíz del proyecto**:
```env
GEMINI_API_KEY=tu_gemini_key
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...

# Configuración opcional
EMBEDDINGS_DEVICE=cpu  # o cuda si tienes GPU
RAG_TOP_K=5           # Número por defecto de fuentes
```

## 🎯 Casos de Uso

- **Investigación legal**: Búsqueda de decretos, leyes y regulaciones específicas con análisis temporal
- **Análisis normativo**: Consulta de cambios en políticas públicas con indicadores de actualidad
- **Estudios académicos**: Investigación en documentos oficiales con contexto histórico
- **Consultoría**: Verificación rápida de información gubernamental con fuentes verificables

## 🔧 Para Desarrolladores

### Estructura Modular
```
chatbot_gradio/
├── config/          → Configuración centralizada y validación
├── core/            → Lógica RAG (database, embeddings, llm, pipeline)
├── interface/       → Componentes UI (chat, rendering, history)
├── chat_styles.css  → Estilos personalizados optimizados
└── app.py          → Punto de entrada principal
```

### Características Técnicas
- **Temporal Reranking**: Boost automático de documentos recientes
- **Embedded History**: Sistema de historial con fuentes integradas
- **Scroll Inteligente**: Auto-scroll con soporte para contenido dinámico
- **Validación de Entorno**: Verificación automática de configuración
- **Logging Estructurado**: Archivo `chatbot_gradio.log` en directorio de trabajo

### Extensibilidad
- Agregar nuevos proveedores LLM en `core/llm_client.py`
- Personalizar UI en `interface/chat_ui.py` y `chat_styles.css`
- Modificar lógica RAG en `core/rag_pipeline.py`
- Ajustar configuración en `config/config.py`

### Desarrollo y Debug
```bash
# Ejecutar con logging detallado
python app.py  # Los logs se guardan en chatbot_gradio.log

# Verificar configuración
python -c "from config.config import validate_environment; print(validate_environment())"
```

---

**Nota**: Este chatbot es para **consulta únicamente**. Para actualizar la base de datos con nuevos documentos DOF, utiliza las herramientas del proyecto principal `dof-rag`.