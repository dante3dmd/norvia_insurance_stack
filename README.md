# Norvia Insurance Stack

## Seguridad de entradas y secretos

La API acepta únicamente archivos PDF, TXT y MD de hasta 10 MiB (`MAX_UPLOAD_SIZE_BYTES`). Rechaza cargas vacías, verifica la firma PDF y valida que TXT/MD sean UTF-8 sin bytes NUL; el MIME persistido se deriva de la extensión admitida y no del header enviado por el cliente.

Los documentos y las preguntas se envían como datos JSON no confiables, mientras que las reglas anti-injection viajan en un mensaje `system` separado. El modelo recibe instrucciones explícitas para no obedecer órdenes contenidas en documentos y citar solo evidencia recuperada. Es una mitigación en capas, no una garantía absoluta contra prompt injection; el flujo no expone herramientas ni acciones externas al modelo.

Las credenciales se configuran mediante variables de entorno (`DATABASE_URL`, `POSTGRES_PASSWORD`, `LITELLM_API_KEY`/`OPENAI_API_KEY`). `.env.example` contiene solo valores de reemplazo, nunca claves reales; sustituye esos placeholders localmente y no subas `.env` al repositorio.

## Observabilidad

La API y el worker exportan trazas OpenTelemetry a Phoenix (`http://localhost:6006` en el stack local). Los logs de aplicación se emiten como JSON e incluyen `trace_id` y `span_id` cuando hay una traza activa, para correlacionar eventos entre servicios.

Las trazas registran latencias de embedding, consulta vectorial y generación del LLM; conteos de resultados, tamaño del contexto y chunks; y tokens reales reportados por el proveedor. En proveedores sin contadores se informa una estimación separada, no un uso real.

Por privacidad, no se adjuntan preguntas, prompts, respuestas ni texto de chunks a spans o logs. La recuperación registra únicamente tamaños, conteos y scores agregados.

## Evaluación mínima de recuperación

Este repositorio incluye una mini evaluación de recuperación para comprobar que el motor vectorial encuentra los fragmentos esperados con un conjunto pequeño y determinista de preguntas.

### Comando de ejecución

Desde la raíz del repositorio:

```powershell
$env:PYTHONPATH = "backend"
python backend/scripts/retrieval_eval.py
```

En Bash / zsh también funciona:

```bash
PYTHONPATH=backend python backend/scripts/retrieval_eval.py
```

### Qué mide la evaluación

La métrica usada es `hit_rate@3`:

- Se toma una batería pequeña de 5 preguntas sobre pólizas y coberturas.
- Se ordenan los fragmentos por similitud con la pregunta.
- Se considera acierto si un fragmento dentro de los 3 primeros contiene los términos del tema esperado; la comparación ignora acentos y conectores como «y».
- El score final es:

```text
hit_rate@3 = aciertos / total_de_preguntas
```

### Resultado esperado

Con la biblioteca sintética usada en esta prueba, el resultado objetivo es igual o superior a `0.8` (al menos 4 de 5 preguntas correctas).

Ejemplo de salida esperada:

```json
{
  "k": 3,
  "n_questions": 5,
  "hits": 5,
  "hit_rate_at_k": 1.0,
  "results": [
    {"question": "¿Qué cubre la póliza en caso de daños materiales del vehículo?", "hit": true},
    {"question": "¿Qué pasa si necesito asistencia en carretera?", "hit": true},
    {"question": "¿Incluye gastos médicos y reembolso por urgencias?", "hit": true},
    {"question": "¿Qué exclusiones aparecen en la póliza?", "hit": true},
    {"question": "¿Durante cuánto tiempo dura la cobertura y cómo funciona el deducible?", "hit": true}
  ]
}
```

### Límites de esta evaluación

- Es una prueba de saneamiento del sistema, no un benchmark estadísticamente significativo.
- El corpus es muy pequeño y sintético, pensado para detectar regresiones simples y comprobar que el pipeline de recuperación funciona.
- No sustituye a una evaluación con documentos reales, más preguntas y un conjunto anotado por dominio.
- La calidad real de la recuperación debe validarse con un dataset más amplio antes de salir a producción.

### Cuándo usarla

Se recomienda ejecutarla en cada cambio relevante del motor de recuperación o del chunking para comprobar que la relevancia sigue siendo estable.
