# Plan canónico: evaluación humana del agente DOF

Estado: arquitectura aprobada e implementación inicial en curso. Este documento
es la fuente canónica del MVP. Los cambios de contrato, persistencia o seguridad
deben actualizarlo en el mismo cambio de código.

## Decisiones vigentes

- El sitio de evaluación vive por completo en `dof-rag`; no depende de
  `dof-rag-website`, Astro, GitHub Pages ni su `base` path.
- Se usa una aplicación Python de mismo origen con Air, HTML progresivo y una
  cola local. Air se fija en `0.35.0`, última versión compatible con el Python
  3.12 administrado por el proyecto; versiones posteriores requieren Python
  3.13.
- Sí se guardan las preguntas y las respuestas. Sin ese par no sería posible
  auditar el feedback, reproducir fallos ni producir candidatos revisables para
  v5. También se guardan evidencia, trazas públicas y procedencia.
- La base de evaluación es SQLite y está separada del corpus, chunks e índices.
- La UI crea una ejecución y consulta su estado. No mantiene abierta durante
  decenas de segundos la petición que recibió la pregunta.
- La recuperación inicial es léxica y funciona sin esperar a que termine el
  índice vectorial.

## Objetivo y alcance

El MVP permitirá que un grupo pequeño de personas formule preguntas reales al
agente del DOF, inspeccione la respuesta y lo que la sustenta, y envíe feedback
estructurado. Debe servir tanto para detectar errores de respuesta como fallos
de recuperación, cobertura, citas y comprensión.

Incluye:

- acceso controlado mediante token de invitación y sesión firmada;
- pregunta, fecha de corte opcional y `required_hops` entre 1 y 5;
- estados visibles `queued`, `running`, `succeeded` y `failed` mediante polling;
- respuesta, citas, advertencias, documentos, pasajes y traza pública;
- feedback append-only con rating, tipos de problema y comentario;
- snapshot por ejecución de código, corpus, chunks, índice, modelo y
  configuración;
- historial reciente del mismo evaluador;
- operación inicial desde la MacBook Pro actual con un solo worker.

## Fuera del MVP

- modificar automáticamente `eval/dof_queries_v4.jsonl` o promover ejemplos a
  v5 sin revisión humana;
- chat con memoria, cuentas públicas, restablecimiento de contraseña o acceso
  autoservicio;
- elegir desde el cliente proveedores, modelos, prompts, bases, `top_k` u otros
  argumentos de herramientas;
- acceso directo del navegador a SQLite;
- streaming token a token, cancelación fuerte de una llamada ya enviada al
  proveedor, múltiples workers o alta disponibilidad;
- búsqueda web o fuentes distintas del corpus DOF;
- integrar la UI en Astro durante el MVP. El sitio público podría enlazar a la
  app más adelante, pero no forma parte de su ruta de ejecución.

## Arquitectura

```text
Navegador
  | HTTPS, HTML/forms/fetch, cookie de sesión de mismo origen
  v
Aplicación Air en dof-rag
  - UI y rutas HTTP en human_eval/app.py
  - sesión, CSRF, validación y límites
  - EvaluationService + cola local, 1 worker
  - SQLite de evaluación separado
  |
  v
AgentRunner + DofToolbox
  - corpus/chunks SQLite abiertos de solo lectura
  - recuperación léxica completa
  - índice vectorial opcional cuando esté completo y versionado
  - proveedor del modelo, con claves solo en variables del backend
```

Air aporta el proceso ASGI y permite iterar la UI junto al controlador. La
lógica de contratos, almacenamiento, cola y ejecución permanece en módulos
pequeños y ajenos a Air; esto reduce el costo de sustituir el framework si su
API, todavía joven, resulta inestable.

No se usan `BackgroundTasks` para ejecutar al agente: son trabajo en proceso y
no sustituyen una cola recuperable. `EvaluationService` responde rápido con un
`run_id`, procesa en su hilo worker y recupera al arrancar las ejecuciones que
quedaron en cola. Una ejecución que estaba iniciada se marca fallida al
reiniciar, porque no puede saberse si la llamada externa terminó.

## Contrato HTTP v1

El contrato público del MVP es de mismo origen. Las rutas HTML usan formularios
URL-encoded, redirects `303` después de escritura y fragmentos HTML para
polling. Los endpoints de salud y capacidades usan JSON. No se habilita CORS.

| Método y ruta | Autenticación | Resultado |
| --- | --- | --- |
| `GET /login` | no | formulario de invitación y CSRF |
| `POST /login` | CSRF + token | crea sesión y redirige a `/` |
| `POST /logout` | sesión + CSRF | destruye la sesión |
| `GET /` | sesión | pregunta nueva e historial propio |
| `POST /runs` | sesión + CSRF | crea/idempotentiza ejecución y redirige |
| `GET /runs/{run_id}` | sesión y propiedad | página completa de la ejecución |
| `GET /runs/{run_id}/status` | sesión y propiedad | fragmento de estado/resultado |
| `POST /runs/{run_id}/feedback` | sesión, propiedad y CSRF | añade feedback y redirige |
| `GET /api/v1/health` | no | salud del proceso y SQLite |
| `GET /api/v1/capabilities` | no | contrato, modo, modelo y límites seguros |

### Crear una ejecución

`POST /runs` acepta exclusivamente:

```text
question           string, 3-2000 caracteres
as_of              fecha ISO YYYY-MM-DD o vacío
required_hops      entero 1-5
client_request_id  identificador opaco 1-128, generado por el formulario
csrf_token         secreto de la sesión
```

El cliente no puede enviar argumentos de herramientas. `client_request_id`
permite que un reenvío del mismo formulario por el mismo evaluador reutilice la
ejecución. Reutilizarlo con otra entrada es conflicto. Hay una sola ejecución
activa por evaluador y una cola global acotada.

### Consultar una ejecución

La página y el fragmento representan cuatro estados:

- `queued`: aceptada, esperando worker;
- `running`: el agente está consultando herramientas o proveedor;
- `succeeded`: respuesta y resultado público persistidos;
- `failed`: código y mensaje públicos estables, sin excepción interna.

Mientras el estado no sea terminal, el navegador espera aproximadamente dos
segundos y solicita `GET /runs/{run_id}/status`. La respuesta terminal se
construye desde el resultado ya persistido, no volviendo a consultar un índice
que pudo cambiar.

El objeto lógico almacenado y presentado contiene:

```json
{
  "run_id": "uuid",
  "status": "succeeded",
  "question": "...",
  "as_of": null,
  "required_hops": 2,
  "created_at": "...",
  "started_at": "...",
  "completed_at": "...",
  "provenance": {
    "code_revision": "git-sha",
    "code_dirty": false,
    "corpus_version": "...",
    "chunker_version": "...",
    "vector_available": false,
    "vector_index_version": null,
    "provider": "openai-responses",
    "model": "...",
    "configuration": {
      "retrieval_mode": "lexical",
      "max_model_turns": 8,
      "max_tool_calls": 8,
      "reasoning_effort": "low"
    }
  },
  "result": {
    "answer": {
      "text": "...",
      "citation_ids": [123],
      "premise_status": "supported"
    },
    "evidence": [
      {"chunk_id": 123, "document_id": 45, "path": "...", "text": "...", "cited": true}
    ],
    "documents": [
      {"document_id": 45, "path": "...", "publication_date": "...", "title": "...", "cited": true}
    ],
    "coverage": {"required": ["..."], "missing": [], "complete": true},
    "verification": {},
    "trace": [],
    "warnings": [],
    "usage": {},
    "elapsed_ms": 12345
  }
}
```

Los fallos usan códigos públicos como `provider_unavailable`, `rate_limited`,
`queue_full` o `internal_error`. Las excepciones y detalles del proveedor solo
se escriben en logs locales.

### Registrar feedback

`POST /runs/{run_id}/feedback` acepta:

```text
rating         helpful | partially_helpful | not_helpful
problem_types  cero o más valores del vocabulario cerrado
comment        string de hasta 2000 caracteres
csrf_token     secreto de la sesión
```

El vocabulario es `incorrect_answer`, `missing_evidence`, `bad_citation`,
`incomplete_coverage`, `cutoff_error`, `hard_to_understand` y `other`. Cada
envío crea un UUID nuevo; nunca reemplaza feedback previo ni modifica la
ejecución o v4.

## Persistencia

La base inicial es `var/human_evaluation.sqlite`, está excluida de Git y no se
comparte con ninguna base del agente. Usa WAL, foreign keys, `busy_timeout` y
una conexión corta por operación.

Tablas fuente:

- `runs`: fila inmutable con pregunta, fecha de corte, `required_hops`, hash de
  evaluador, idempotencia y snapshot JSON de procedencia;
- `run_events`: log append-only con secuencia y eventos `queued`, `started`,
  `succeeded` o `failed`; el payload terminal guarda la respuesta exacta o el
  error público;
- `feedback`: filas append-only con UUID, ejecución, rating, etiquetas,
  comentario y timestamp;
- `schema_meta`: versión del esquema.

Guardar pregunta y respuesta es deliberado. El feedback aislado carece de
contexto y no permite distinguir un error del modelo de un cambio posterior del
índice. También se guarda el resultado público completo: documentos, pasajes,
citas, cobertura, verificación, traza, uso y duración. No se guardan tokens de
invitación, cookies, claves de proveedor, cabeceras ni razonamiento privado.

El hash del token identifica al evaluador para propiedad, límites e
idempotencia, pero no se presenta como identidad. Antes de un piloto más amplio
debe definirse retención y borrado administrativo de preguntas que puedan
contener datos personales. Esa futura operación será explícita y auditable; no
forma parte del flujo normal append-only.

Una exportación administrativa futura puede unir entrada, resultado, evidencia
y feedback para generar candidatos v5. Un humano debe corregir, deduplicar y
aprobar cada candidato antes de incorporarlo a un dataset versionado.

## Citas, evidencia y trazas

- Solo un chunk devuelto por `read_chunks` puede convertirse en cita.
- Cada `chunk_id` citado se resuelve al texto persistido y a su documento.
- La UI enlaza los IDs citados con pasajes expandibles y distingue documento
  consultado, usado como evidencia y citado.
- Se muestran búsquedas, documentos considerados, chunks leídos,
  verificaciones, límites y tiempos que sean seguros para el evaluador.
- No se muestra chain-of-thought, mensajes privados del proveedor, claves,
  cabeceras ni configuración de clientes.
- `invalid_citations`, fallos de herramienta, `stop_reason` y cobertura faltante
  aparecen como advertencias visibles.
- Una búsqueda no cuenta como evidencia; el pasaje leído sí.

## Preguntas multidocumento y `required_hops`

El usuario puede indicar de 1 a 5 documentos mínimos, no IDs concretos. La UI
explica que 2 o más sirve para comparaciones o preguntas que requieren fuentes
distintas. El backend pasa el valor validado a `AgentRunner`.

El resultado separa:

- documentos requeridos, leídos y citados;
- requisitos explícitos inferidos de la pregunta, como años o publicaciones;
- requisitos faltantes y `coverage.complete`;
- causa de terminación y advertencias.

Una ejecución no puede declararse completa para `required_hops=2` sin citas que
cubran al menos dos documentos distintos. Si alcanza límites antes de cubrir la
pregunta, la respuesta se conserva para diagnóstico y se rotula como cobertura
incompleta; continúa siendo evaluable.

## Seguridad, autenticación y límites

- Claves y configuración de proveedores existen solo en variables de entorno
  del backend.
- `DOF_EVALUATOR_TOKENS` contiene invitaciones individuales. En login se
  comparan hashes con tiempo constante. El token crudo no se persiste ni se
  devuelve.
- La cookie contiene un hash de evaluador y CSRF dentro de una sesión firmada;
  una firma no cifra contenido, por lo que tampoco se colocan secretos de
  proveedor en la sesión.
- La cookie es `HttpOnly`, `SameSite=Lax`, tiene vencimiento y debe activar
  `Secure` (`DOF_SECURE_COOKIE=true`) detrás de HTTPS.
- Toda escritura, incluido login, valida CSRF. Las páginas y JSON llevan
  `Cache-Control: no-store`, CSP, `nosniff` y política de referrer.
- `TrustedHostMiddleware` usa una lista explícita configurada para el host del
  túnel. La app no habilita CORS porque UI y backend comparten origen.
- Los cuerpos tienen un límite inicial de 16 KiB; contratos validan longitud,
  fechas, enums y campos. El navegador nunca controla rutas de bases o
  parámetros arbitrarios.
- Límites iniciales: una ejecución activa por evaluador, diez creaciones por
  hora, cola global de veinte y un worker. Turnos y llamadas a herramientas
  también están acotados en el backend.
- Las ejecuciones solo son visibles para el hash de evaluador propietario. Los
  endpoints públicos de salud/capacidades no incluyen rutas locales ni secretos.
- Los logs usan `run_id` y no incluyen tokens ni cuerpos completos por defecto.

## Despliegue previsto

El MVP se ejecutará en la MacBook Pro actual, ligado inicialmente a
`127.0.0.1:8765`. La UI y el backend se publican como una sola app ASGI. Para
pruebas humanas remotas se colocará delante un túnel o reverse proxy HTTPS que
termine TLS, limite cuerpos y use un hostname estable; todavía debe elegirse el
proveedor.

Configuración mínima, con valores de ejemplo que no deben guardarse en Git:

```bash
export DOF_EVALUATOR_TOKENS='token-individual-1,token-individual-2'
export DOF_SESSION_SECRET='valor-aleatorio-de-al-menos-32-caracteres'
export DOF_ALLOWED_HOSTS='localhost,127.0.0.1,piloto.example'
export DOF_SECURE_COOKIE='true'
export DOF_AGENT_PROVIDER='openai-responses'
export DOF_AGENT_MODEL='modelo-configurado-en-backend'
export OPENAI_API_KEY='...'
uv run python -m human_eval.app
```

La recuperación por defecto es léxica. El worker único evita competir
agresivamente con la indexación en curso. Antes del piloto externo faltan el
supervisor local, el túnel HTTPS y un procedimiento de backup de
`var/human_evaluation.sqlite`; el corpus y los índices siguen siendo
dependencias de solo lectura con su propio ciclo de respaldo.

## Higiene de reproducibilidad

- `scripts/eval_v4_full.py` es código de evaluación, no un resultado generado,
  y debe versionarse.
- `reports/eval_v4_retrieval.md` documenta metodología y procedencia canónicas,
  por lo que también debe versionarse.
- JSON de corridas, caches, listas de fallos, logs, bases, WAL/SHM, checkpoints
  y archivos rankeados son artefactos generados: se conservan localmente y se
  excluyen de Git cuando su patrón es inequívoco.
- Planes y reportes que expresen decisiones humanas no se eliminan ni se ignoran
  por un patrón amplio.
- v4 permanece congelado. El feedback solo alimentará una exportación de
  candidatos que pueda revisarse para v5.

## Fases y criterios de aceptación

### Fase 0 — contrato y reproducibilidad

- El presente documento describe una sola app en `dof-rag` y no requiere Astro.
- Código, reportes canónicos y resultados generados quedan clasificados sin
  eliminar artefactos existentes.
- Cada ejecución declara qué versiones y configuración deben capturarse.

### Fase 1 — núcleo y almacenamiento

- Crear, consultar y evaluar una ejecución funciona con ejecutor falso, sin red
  ni corpus.
- SQLite sobrevive al reinicio y usa eventos/feedback append-only.
- Pregunta, respuesta exacta y procedencia quedan persistidas.
- Idempotencia y propiedad están aisladas por evaluador.

### Fase 2 — sitio Air mínimo

- Login intercambia una invitación por sesión sin persistir el token.
- El formulario acepta pregunta, fecha y hops; muestra progreso mediante
  polling y no bloquea una petición larga.
- Respuesta, citas, documentos, pasajes, cobertura y traza son legibles con
  teclado y en móvil.
- Feedback estructurado confirma que fue guardado y que no modifica v4.
- Pruebas verifican sesión, CSRF, aislamiento, persistencia y endpoints seguros.

### Fase 3 — integración real del agente

- Una pregunta léxica real termina y devuelve citas resolubles y evidencia.
- Una pregunta con `required_hops=2` no se declara completa sin dos documentos
  citados.
- Fallos y límites producen un estado terminal y un mensaje público estable.
- Una prueba de humo confirma que corpus/chunks permanecen de solo lectura.

### Fase 4 — piloto controlado

- HTTPS, hostname, cookie `Secure`, tokens individuales y límites se validan
  desde una red externa.
- Se prueba reinicio, backup/restauración y recuperación de cola.
- Se acuerdan consentimiento, retención y contacto para reportar problemas.
- Una exportación administrativa produce candidatos v5 sin tocar v4.

## Riesgos y decisiones abiertas

- Air sigue evolucionando y la versión compatible con Python 3.12 no es la más
  reciente. Se debe decidir después del piloto si migrar todo el proyecto a
  Python 3.13, mantener 0.35 o sustituir solo la capa web.
- Exponer la MacBook requiere elegir túnel/proxy, dominio, supervisor y política
  de actualización antes de invitar evaluadores.
- La cola y el rate limit son locales y se reinician con el proceso; son
  suficientes para un piloto de un nodo, no para varios procesos.
- Deben fijarse presupuesto por modelo, timeout efectivo y respuesta ante cuota
  agotada.
- La huella del índice vectorial debe ser verificable antes de activar modo
  híbrido; “el archivo existe” no basta como versión de producción.
- Falta definir retención y eliminación administrativa de preguntas,
  comentarios e IPs (idealmente las IPs no se persisten).
- Debe decidirse si los documentos enlazan al DOF oficial, a una vista local
  sanitizada o únicamente al pasaje persistido.
- Para público general quizá convenga inferir `required_hops`; durante el MVP se
  mantiene visible para estudiar si los evaluadores lo comprenden.
