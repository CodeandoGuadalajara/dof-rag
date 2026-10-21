# Lector de pasajes (#89 / #91)

Primera entrega compartida: `/pasajes/{chunk_id}#pasaje-{chunk_id}` abre un
pasaje del corpus y hasta un vecino a cada lado, dentro del mismo documento.
Por ejemplo: `/pasajes/1#pasaje-1` (si ese ID existe en el corpus configurado).

- Usa `DOF_CORPUS_DB` y `DOF_CHUNKS_DB`, igual que el ejecutor del agente.
- Reutiliza `DofRetriever.read_chunks` y su verificación de hashes. No inicia
  modelos, embeddings ni ejecuciones de preguntas; no lee chats ni sus permisos.
- Es público porque muestra exclusivamente publicaciones del corpus, nunca
  texto de conversaciones. Abrir el enlace no acredita una cita como verificada.
- Muestra título disponible, fecha de publicación, sección y advertencias de
  extracción y vigencia. El enlace oficial lleva a las **ediciones de la fecha**;
  el índice actual no proporciona una URL oficial específica por documento.
- ID ausente: 404. Error de lectura o integridad: 503 sin mostrar texto no verificado
  ni detalles internos. El contexto es acotado; no es un lector de documento completo.

Pendiente: búsqueda con URL compartible (#91), integración de citas verificadas
con Chat/Preguntas (#89), y revisión visual de móvil/teclado antes de desplegar.
No se modificaron los enlaces ni la visibilidad actual de las conversaciones.

Pruebas locales, sin inferencia:

```sh
.venv/bin/python -m unittest tests.test_passage_reader tests.test_human_eval -q
```
