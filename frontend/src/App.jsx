import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const statusStyles = {
  pending: 'bg-slate-500/15 text-slate-200 ring-1 ring-slate-400/40',
  processing: 'bg-amber-500/15 text-amber-200 ring-1 ring-amber-400/40',
  completed: 'bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-500/40',
  failed: 'bg-rose-500/15 text-rose-200 ring-1 ring-rose-500/40',
};

// Fetch the latest document list and refresh it every 3 seconds to keep the ingest panel live.
async function fetchDocuments() {
  const response = await fetch(`${API_BASE}/api/v1/documents`);
  if (!response.ok) {
    throw new Error('No se pudo recuperar la lista de documentos');
  }
  return response.json();
}

// Extract a human-readable message from either string or structured API error details.
function getApiErrorMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === 'string') return detail;
  if (typeof detail?.message === 'string') return detail.message;
  return fallback;
}

// Upload a selected PDF to the FastAPI ingest endpoint and return the server response.
async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/api/v1/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(getApiErrorMessage(payload, 'Error al subir el documento'));
  }

  return response.json();
}

// Build a compact display label for the document status and its ready-to-render style class.
function getStatusPill(status) {
  return {
    label: status?.toUpperCase?.() || 'UNKNOWN',
    className: statusStyles[status] || 'bg-slate-500/15 text-slate-200 ring-1 ring-slate-400/40',
  };
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [question, setQuestion] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatResult, setChatResult] = useState(null);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // Refresh the document table on mount and every 3 seconds while the user is in the dashboard.
  useEffect(() => {
    let ignore = false;

    const syncDocuments = async () => {
      try {
        const data = await fetchDocuments();
        if (!ignore) {
          setDocuments(data);
          setLoadingDocuments(false);
        }
      } catch (error) {
        if (!ignore) {
          setErrorMessage(error.message || 'No se pudo cargar la lista');
          setLoadingDocuments(false);
        }
      }
    };

    syncDocuments();
    const intervalId = setInterval(syncDocuments, 3000);

    return () => {
      ignore = true;
      clearInterval(intervalId);
    };
  }, []);

  // Toggle a selected document so the chat can filter retrieval to specific document_ids.
  const toggleDocumentSelection = (documentId) => {
    setSelectedDocumentIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  };

  // Handle drag-and-drop uploads and immediately refresh the document table afterward.
  const handleFiles = async (incomingFiles) => {
    if (!incomingFiles?.length) return;

    setUploading(true);
    setErrorMessage('');

    try {
      const results = await Promise.all(
        Array.from(incomingFiles).map(async (file) => {
          if (!file.name.toLowerCase().endsWith('.pdf')) {
            return { error: 'Solo se admiten archivos PDF.' };
          }
          return uploadDocument(file);
        }),
      );

      const uploadErrors = results.filter((item) => item?.error);
      if (uploadErrors.length > 0) {
        setErrorMessage(uploadErrors[0].error);
      }

      const refreshed = await fetchDocuments();
      setDocuments(refreshed);
    } catch (error) {
      setErrorMessage(error.message || 'No se pudo completar la subida');
    } finally {
      setUploading(false);
      setDragActive(false);
    }
  };

  // Send the chat question to the RAG API and render the returned structured answer.
  const handleChatSubmit = async (event) => {
    event.preventDefault();
    if (!question.trim()) return;

    setChatLoading(true);
    setErrorMessage('');

    try {
      const response = await fetch(`${API_BASE}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          document_ids: selectedDocumentIds,
          k: 5,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(getApiErrorMessage(payload, 'La consulta no pudo completarse'));
      }

      const payload = await response.json();
      setChatResult(payload);
      setQuestion('');
    } catch (error) {
      setErrorMessage(error.message || 'No se pudo completar la respuesta de chat');
      setChatResult(null);
    } finally {
      setChatLoading(false);
    }
  };

  const selectedDocsLabel = useMemo(() => {
    if (selectedDocumentIds.length === 0) return 'Todos los documentos';
    return `${selectedDocumentIds.length} documentos seleccionados`;
  }, [selectedDocumentIds]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 antialiased">
      <div className="mx-auto max-w-7xl p-6 lg:p-10">
        <header className="mb-8 flex flex-col gap-3 border-b border-slate-800 pb-6 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.25em] text-cyan-400">Norvia Insurance</p>
            <h1 className="mt-2 text-3xl font-semibold text-white">Panel de ingestión y RAG</h1>
          </div>
          <div className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-xs font-medium text-cyan-200">
            Poll cada 3s
          </div>
        </header>

        <div className="grid gap-6 xl:grid-cols-[1.1fr,0.9fr]">
          <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-glow">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Administración de documentos</h2>
              <span className="text-xs text-slate-400">{documents.length} documentos</span>
            </div>

            <div
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(event) => {
                event.preventDefault();
                handleFiles(event.dataTransfer.files);
              }}
              className={`mb-5 rounded-2xl border-2 border-dashed p-6 text-center transition ${
                dragActive ? 'border-cyan-400 bg-cyan-500/10' : 'border-slate-700 bg-slate-950/40'
              }`}
            >
              <input
                type="file"
                accept="application/pdf"
                className="hidden"
                id="document-upload-input"
                onChange={(event) => handleFiles(event.target.files)}
              />
              <label htmlFor="document-upload-input" className="cursor-pointer">
                <div className="text-lg font-medium text-white">Arrastra y suelta PDFs aquí</div>
                <div className="mt-2 text-sm text-slate-400">o haz clic para seleccionar archivos</div>
              </label>
              {uploading && <div className="mt-3 text-sm text-cyan-300">Subiendo documentos…</div>}
            </div>

            {errorMessage && (
              <div className="mb-4 rounded-xl border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                {errorMessage}
              </div>
            )}

            <div className="overflow-hidden rounded-xl border border-slate-800">
              <div className="max-h-[440px] overflow-auto">
                <table className="min-w-full divide-y divide-slate-800 text-left">
                  <thead className="bg-slate-800/70 text-xs uppercase tracking-wide text-slate-300">
                    <tr>
                      <th className="px-3 py-3">Filtrar</th>
                      <th className="px-3 py-3">Archivo</th>
                      <th className="px-3 py-3">Estado</th>
                      <th className="px-3 py-3">Tamaño</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 bg-slate-950/30">
                    {loadingDocuments ? (
                      <tr>
                        <td colSpan="4" className="px-3 py-6 text-center text-sm text-slate-400">
                          Cargando documentos…
                        </td>
                      </tr>
                    ) : documents.length === 0 ? (
                      <tr>
                        <td colSpan="4" className="px-3 py-6 text-center text-sm text-slate-400">
                          No hay documentos aún.
                        </td>
                      </tr>
                    ) : (
                      documents.map((document) => {
                        const pill = getStatusPill(document.status);
                        return (
                          <tr key={document.id} className="hover:bg-slate-800/40">
                            <td className="px-3 py-3">
                              <input
                                type="checkbox"
                                checked={selectedDocumentIds.includes(document.id)}
                                onChange={() => toggleDocumentSelection(document.id)}
                                className="h-4 w-4 rounded border-slate-600 bg-slate-900 text-cyan-500 focus:ring-cyan-500"
                              />
                            </td>
                            <td className="px-3 py-3">
                              <div className="font-medium text-white">{document.filename}</div>
                              <div className="text-xs text-slate-400">{document.content_type}</div>
                              {document.status === 'failed' && document.error_message && (
                                <div className="mt-1 max-w-xs text-xs text-rose-300">{document.error_message}</div>
                              )}
                            </td>
                            <td className="px-3 py-3">
                              <span
                                className={`inline-flex rounded-full px-2.5 py-1 text-[10px] font-semibold ${pill.className}`}
                                title={document.error_message || 'Sin errores'}
                              >
                                {pill.label}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-sm text-slate-300">
                              {(document.file_size / 1024).toFixed(1)} KB
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-glow">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Chat RAG</h2>
              <div className="rounded-full border border-slate-700 bg-slate-800 px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-300">
                {selectedDocsLabel}
              </div>
            </div>

            <form onSubmit={handleChatSubmit} className="space-y-4">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows={5}
                placeholder="Pregunta sobre las pólizas, coberturas, exclusiones o condiciones..."
                className="w-full rounded-xl border border-slate-700 bg-slate-950/50 px-3 py-3 text-sm text-white placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
              />

              <div className="flex items-center justify-between gap-3">
                <div className="text-xs text-slate-400">Filtra por {selectedDocumentIds.length || 'todos'} documento(s)</div>
                <button
                  type="submit"
                  disabled={chatLoading || !question.trim()}
                  className="inline-flex items-center justify-center rounded-xl bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                >
                  {chatLoading ? 'Consultando…' : 'Enviar pregunta'}
                </button>
              </div>
            </form>

            {chatResult && (
              <div className="mt-6 space-y-4">
                {!chatResult.has_sufficient_evidence && (
                  <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                    No hay evidencia suficiente para responder con seguridad. La respuesta fue rechazada por el backend.
                  </div>
                )}

                <div className="rounded-xl border border-slate-700 bg-slate-950/30 p-4">
                  <div className="mb-2 text-xs uppercase tracking-[0.2em] text-slate-400">Respuesta</div>
                  <p className="whitespace-pre-wrap text-sm leading-7 text-slate-100">{chatResult.answer}</p>
                </div>

                {chatResult.citations?.length > 0 && (
                  <div className="rounded-xl border border-slate-700 bg-slate-950/30 p-4">
                    <div className="mb-3 text-xs uppercase tracking-[0.2em] text-slate-400">Citas</div>
                    <div className="flex flex-wrap gap-2">
                      {chatResult.citations.map((citation, index) => (
                        <button
                          key={`${citation.document_id}-${index}`}
                          type="button"
                          onClick={() => setSelectedCitation(citation)}
                          className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-200 transition hover:bg-cyan-500/20"
                        >
                          Documento {citation.document_id} · Página {citation.page ?? 'N/A'}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </div>

      {selectedCitation && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm">
          <div className="w-full max-w-2xl rounded-2xl border border-slate-700 bg-slate-900 p-5 shadow-glow">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-cyan-400">Cita</p>
                <h3 className="mt-1 text-xl font-semibold text-white">Documento {selectedCitation.document_id}</h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedCitation(null)}
                className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-sm text-slate-200 hover:bg-slate-800"
              >
                Cerrar
              </button>
            </div>

            <div className="space-y-3 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
              <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate-400">
                <span>Página {selectedCitation.page ?? 'N/A'}</span>
                <span>Score {Number(selectedCitation.score ?? 0).toFixed(3)}</span>
              </div>
              <p className="whitespace-pre-wrap text-sm leading-7 text-slate-200">{selectedCitation.chunk}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
