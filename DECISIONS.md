
# Decisiones de arquitectura

## 1. Estrategia de Chunking y Tamaño de Contexto

### 1.1 Extracción
• Decisión: Usar PyMuPDF4LLM para llevar los PDF a markdown.
• Justificación: PyMuPDF4LLM posee análisis de estructural del documento y convierte el PDF a Markdown respetando tablas, títulos y listas. 

### 1.2 Chunking
• Decisión: MarkdownNodeParser de LlamaIndex para los archivos Markdown y HierarchicalNodeParser + SentenceSplitter para los archivos txt
• Justificación: Esto garantiza que las tablas no se rompan y se mantenga la jerarquía visual para los archivos markdown y para los txt es vital usar la relación Padre-Hijo (Parent-Child) donde el sistema busca la cláusula exacta (hijo de 512 tokens), pero le envía al LLM el artículo o sección completa (padre de 2048 tokens) para que no pierda el contexto de la póliza. 

## 2. Política de Rechazo (Evitar Alucinaciones)

• Decisión: Validación en dos capas: Umbral de similitud vectorial estricto (Distancia Coseno \(\le \) 0.35 / Similitud \(\ge \) 0.65) + Validación Estructurada con Pydantic (IsEvidenceSufficient: bool) en el LLM.
• Justificación: Si los fragmentos recuperados por pgvector tienen un score de similitud muy bajo, el sistema ni siquiera invoca al LLM y rechaza por falta de contexto. Si pasan el filtro, se le pasa al LLM un prompt con la instrucción del sistema: "Responde ÚNICAMENTE con el contexto provisto. Si el contexto no contiene la respuesta exacta, marca has_sufficient_evidence: false y devuelve el texto de rechazo configurado". El backend validará este JSON con Pydantic v2.

## 3. Aislamiento de Datos (Multi-tenancy)

• Decisión: Modelo de Documentos Globales / Compartidos a nivel de Organización, con control de acceso por document_id o policy_id en las consultas vectoriales.
• Justificación: En una aseguradora, los condicionados generales y pólizas son los mismos para todos los agentes de atención. Dejar los documentos como "globales para la plataforma" simplifica el MVP. Sin embargo, para cumplir con el requisito de filtrado, cada query a pgvector incluirá obligatoriamente un filtro compuesto: WHERE policy_id = ANY(:policy_ids). Se deja fuera del alcance la privacidad estricta por agente de atención.

## 4. Reparto de Responsabilidades: Redis vs. PostgreSQL

• Decisión:
	• Redis: Mensajería de Celery (Broker), Almacenamiento temporal de Rate Limiting (límites de tasa por usuario) y Caché efímera del estado de la tarea.
	• PostgreSQL: Estado de Ingesta Persistente (pending, processing, completed, failed), metadatos del documento, fragmentos (chunks) y embeddings vectoriales.
• Justificación si Redis se cae a mitad de ingesta: Si Redis se cae, Celery perderá el mensaje en tránsito si no está configurado con Late Acknowledgment. Sin embargo, como el estado real del documento vive en PostgreSQL, un mecanismo de conciliación o el reintento del usuario puede verificar que el documento quedó en estado processing o pending en la base de datos SQL y reencolarlo de forma segura. La idempotencia se garantiza verificando el Hash SHA-256 del archivo en Postgres antes de procesarlo.


## 5. Autenticación simple
Decisión: Autenticación por API Key
Justificación: Dado que el MVP integra componentes desacoplados en un monorepo (Frontend en React, Backend HTTP, y un Worker de Celery) y requiere trazabilidad de extremo a extremo con OpenTelemetry, la API Key simplifica drásticamente la propagación del contexto y la validación en segundo plano sin la sobrecarga de gestionar la expiración o el estado de los tokens de sesión.

Para mitigar la única desventaja de la API Key (la seguridad en el cliente React), imita el comportamiento de un token de sesión corto: no permitas que el usuario configure o vea su API Key de forma permanente. Cuando el agente inicie sesión en el MVP, el backend valida sus credenciales y le entrega la API Key, la cual React almacenará únicamente en el sessionStorage (se borra al cerrar la pestaña). El frontend enviará esta clave en el header Authorization: ApiKey <clave> en cada petición.


## 6. Versión de Python
Decisión: Usar la versión 3.12.8
Justificación: Es bien compatible con OpenTelemetry y tiene mejoras en el procesamiento asíncrono

## 7. Versiones para la interfaz web
Decisión: React 18.3.1 y Vite 5.4.10
Justificación: Es el estándar dorado actual en la industria para el desarrollo de Single Page Applications rápidas y ligeras

## 8. Tasas para rate limit
Decisión: 
- Por usuario 5 - 10 por minuto
- Ráfaga 2 - 3 solicitudes
- Global/Aplicación 20 - 50 por minuto
Justificación: 
- Por usuario: Un agente tarda al menos 15-30 segundos en leer una respuesta del LLM y validar las citas del PDF. No puede consumir físicamente más que esto de forma legítima.
- Ráfaga: Permite que si el usuario da "Enter" dos veces por error, o reformula rápido, no sea bloqueado inmediatamente (se maneja con algoritmos como Token Bucket).
- Global/Aplicación: Protege tu billetera y evita alcanzar el Rate Limit de las API de OpenAI/Anthropic (TPM/RPM) en entornos de prueba de menos de 10 personas.

## 9. Telemetría
Decisión: Phoenix
Justificación: Tiene interfaz y la carga mayor del stack recae sobre el RAG y no en la telemetría

## 10. Elección del índice para pgvector entre HNSW y IVFFlat
Decisión: Phoenix
Justificación: HNSW porque es crítica la precisión para mitigar al máximo las alucinaciones. Demora en indexar grandes volúmenes, pero agrega sin necesidad de reconstrucción de índices. Es un grafo que establece la jerarquía y conecta mejor la información
