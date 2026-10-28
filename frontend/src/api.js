const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `API ${response.status}: ${path}`);
  }
  return response.json();
}

function predictionFromApi(payload) {
  const prediction = payload.prediction;
  const review = payload.review;
  return {
    label: prediction.label,
    probabilityBug: prediction.probability_bug,
    classificationThreshold: prediction.classification_threshold,
    predictionId: payload.request_id,
    modelName: prediction.model_name,
    modelVersion: prediction.model_version,
    inferenceDevice: prediction.inference_device,
    reviewRequired: review.required,
    reviewReasons: review.reasons,
    lowConfidenceCheck: review.low_confidence_check,
    oodCheck: review.ood_check,
  };
}

export async function getDatasetStats() {
  return request('/api/v1/dataset-stats');
}

export async function getBaselineReport() {
  return request('/api/v1/baseline-report');
}

export async function getProjectStatus() {
  return request('/api/v1/project-status');
}

export async function predictIssue(form) {
  const data = await request('/api/v1/predictions', { method: 'POST', body: JSON.stringify(form) });
  return predictionFromApi(data);
}

export async function predictIssueFile(file) {
  const query = new URLSearchParams({ filename: file.name });
  const response = await fetch(`${API_BASE}/api/v1/predictions/upload?${query}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.detail ?? `API ${response.status}: file inference failed`);
  return body;
}

export async function getAuditOverview() {
  return request('/api/v1/audit-overview');
}

export async function getAuditCases({ page = 1, pageSize = 10, source = 'all', changedOnly = false, search = '' } = {}) {
  const query = new URLSearchParams({ page, pageSize, source, changedOnly, search });
  return request(`/api/v1/audit-cases?${query}`);
}

export async function getAuditCaseById(id) {
  return request(`/api/v1/audit-cases/${encodeURIComponent(id)}`);
}

export async function getModelComparison() {
  return request('/api/v1/model-comparison');
}
