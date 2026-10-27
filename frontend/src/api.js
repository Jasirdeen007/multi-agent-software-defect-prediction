const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

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

const demoDatasetStats = {
  totalIssues: 926461,
  baselineRows: 926321,
  removedShortTextRows: 140,
  totalProjects: 79,
  sources: [
    { name: 'Bugzilla', count: 378941 },
    { name: 'GitHub', count: 323870 },
    { name: 'JIRA', count: 223650 },
  ],
  classDistribution: [
    { label: 'bug', count: 648841, percent: 70.03 },
    { label: 'non-bug', count: 277620, percent: 29.97 },
  ],
  completeness: [
    { source: 'Bugzilla', severity: true, priority: true, component: true, resolution: true },
    { source: 'GitHub', severity: false, priority: false, component: false, resolution: false },
    { source: 'JIRA', severity: false, priority: true, component: true, resolution: true },
  ],
  topProjects: [
    { project: 'Mozilla Firefox', count: 143882 },
    { project: 'Visual Studio Code', count: 121405 },
    { project: 'Apache Kafka', count: 98214 },
    { project: 'Eclipse Platform', count: 87390 },
    { project: 'Chromium', count: 76840 },
    { project: 'Apache Spark', count: 62470 },
    { project: 'Kubernetes', count: 54800 },
    { project: 'Jenkins', count: 43990 },
  ],
};

const demoBaselineReport = {
  status: 'completed',
  labelSource: 'Original BugHub source labels',
  model: 'DeBERTa-v3-small frozen embeddings + XGBoost',
  inputFeatures: ['model_text'],
  trackingFields: ['source', 'project', 'issue_id'],
  target: 'original_label',
  split: { train: 648424, validation: 138948, test: 138949, randomSeed: 42 },
  threshold: 0.52,
  metrics: {
    accuracy: 0.7852,
    balancedAccuracy: 0.7044,
    macroF1: 0.7195,
    rocAuc: 0.8301,
    mcc: 0.4547,
    brierScore: 0.1483,
  },
};

const demoAuditOverview = {
  status: 'planned',
  totalIssues: 926461,
  auditSampleSize: 5000,
  labelsChanged: 0,
  labelChangeRate: 0,
  agentAgreement: { allAgree: 0, twoOfThree: 0, noMajority: 0 },
  bySource: [
    { source: 'Bugzilla', audited: 0, changed: 0, changeRate: 0 },
    { source: 'GitHub', audited: 0, changed: 0, changeRate: 0 },
    { source: 'JIRA', audited: 0, changed: 0, changeRate: 0 },
  ],
  changeDirection: [
    { from: 'Bug', to: 'Not a Bug', count: 0 },
    { from: 'Not a Bug', to: 'Bug', count: 0 },
  ],
};

const demoCases = [
  { id: 'case_0001', source: 'GitHub', project: 'vscode', issueId: '198765', title: 'Search results disappear after changing folders', originalLabel: 'Bug', auditedLabel: 'Bug', changed: false, judgeConfidence: 0.88, agentAgreement: 'Demo case' },
  { id: 'case_0002', source: 'Bugzilla', project: 'firefox', issueId: '1901043', title: 'Unable to open a downloaded PDF file', originalLabel: 'Non-Bug', auditedLabel: 'Bug', changed: true, judgeConfidence: 0.81, agentAgreement: 'Demo case' },
  { id: 'case_0003', source: 'JIRA', project: 'spark', issueId: 'SPARK-48710', title: 'Job fails when reading a partitioned parquet file', originalLabel: 'Non-Bug', auditedLabel: 'Bug', changed: true, judgeConfidence: 0.84, agentAgreement: 'Demo case' },
];

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
    reviewRequired: review.required,
    reviewReasons: review.reasons,
    lowConfidenceCheck: review.low_confidence_check,
    oodCheck: review.ood_check,
  };
}

export async function getDatasetStats() {
  const data = await request('/api/v1/dataset-stats').catch(() => null);
  if (data) return data;
  await wait(250);
  return demoDatasetStats;
}

export async function getBaselineReport() {
  return request('/api/v1/baseline-report');
}

export async function predictIssue(form) {
  const data = await request('/api/v1/predictions', { method: 'POST', body: JSON.stringify(form) });
  return predictionFromApi(data);
}

export async function getAuditOverview() {
  const data = await request('/api/v1/audit-overview').catch(() => null);
  if (data) return data;
  await wait(250);
  return demoAuditOverview;
}

export async function getAuditCases({ page = 1, pageSize = 10, source = 'all', changedOnly = false, search = '' } = {}) {
  const query = new URLSearchParams({ page, pageSize, source, changedOnly, search });
  const data = await request(`/api/v1/audit-cases?${query}`).catch(() => null);
  if (data) return data;
  await wait(250);
  const term = search.trim().toLowerCase();
  const filtered = demoCases.filter((item) => (source === 'all' || item.source === source) && (!changedOnly || item.changed) && (!term || item.title.toLowerCase().includes(term)));
  return { total: filtered.length, page, pageSize, cases: filtered.slice((page - 1) * pageSize, page * pageSize) };
}

export async function getAuditCaseById(id) {
  const data = await request(`/api/v1/audit-cases/${id}`).catch(() => null);
  if (data) return data;
  await wait(200);
  const item = demoCases.find((caseItem) => caseItem.id === id);
  if (!item) return null;
  return {
    ...item,
    description: 'Demonstration audit detail. Real audit records will appear after selective sampling, KG construction, and agent execution.',
    agents: [
      { name: 'Policy Agent', verdict: item.auditedLabel, confidence: 0.82, reason: 'Evaluates the report against defect policy.' },
      { name: 'Data Agent', verdict: item.originalLabel, confidence: 0.72, reason: 'Checks metadata consistency.' },
      { name: 'Pattern Agent', verdict: item.auditedLabel, confidence: 0.79, reason: 'Reviews defect language and similar reports.' },
    ],
    knowledgeGraphEvidence: ['KG evidence placeholder until the graph stage is implemented.'],
    judge: { finalLabel: item.auditedLabel, confidence: item.judgeConfidence, rationale: 'Demonstration decision for dashboard flow.' },
  };
}

export async function getModelComparison() {
  return request('/api/v1/model-comparison');
}
