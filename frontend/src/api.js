const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? '';
const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function request(path, options) {
  if (!API_BASE) return null;
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) throw new Error(`API ${response.status}: ${path}`);
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
  return {
    label: payload.model_prediction,
    confidence: Math.round(payload.model_confidence * 100),
    predictionId: payload.prediction_id,
    modelVersion: payload.model_version,
    summary: payload.prediction_summary,
    reviewRequired: payload.human_review_required,
    topFeatures: payload.shap_values.map((item) => ({
      feature: item.feature,
      weight: Math.abs(item.weight),
      direction: item.direction,
    })),
  };
}

export async function getDatasetStats() {
  const data = await request('/api/v1/dataset-stats').catch(() => null);
  if (data) return data;
  await wait(250);
  return demoDatasetStats;
}

export async function getBaselineReport() {
  const data = await request('/api/v1/baseline-report').catch(() => null);
  if (data) return data;
  await wait(250);
  return demoBaselineReport;
}

export async function predictIssue(form) {
  const data = await request('/api/v1/predictions', { method: 'POST', body: JSON.stringify(form) }).catch(() => null);
  if (data) return predictionFromApi(data);
  await wait(450);
  const input = `${form.title} ${form.description}`.toLowerCase();
  const defectTerms = /crash|error|fail|broken|exception|regression|incorrect|unable|freeze|cannot/.test(input);
  return {
    label: defectTerms ? 'bug' : 'non-bug',
    confidence: defectTerms ? 84 : 61,
    modelVersion: 'dashboard-preview',
    summary: 'Preview response. Connect FastAPI to local model artifacts for production inference.',
    reviewRequired: !defectTerms,
    topFeatures: defectTerms
      ? [{ feature: 'defect language', weight: 0.38, direction: 'bug' }, { feature: 'reproduction detail', weight: 0.21, direction: 'bug' }, { feature: 'failure wording', weight: 0.16, direction: 'bug' }]
      : [{ feature: 'request wording', weight: 0.31, direction: 'non-bug' }, { feature: 'workflow phrasing', weight: 0.22, direction: 'non-bug' }, { feature: 'low defect signal', weight: 0.14, direction: 'non-bug' }],
  };
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
  const data = await request('/api/v1/model-comparison').catch(() => null);
  if (data) return data;
  await wait(250);
  return {
    setup: { baselineLabelSource: 'Original BugHub source labels', treatmentLabelSource: 'Agent-audited labels (pending)', testSet: 'Held-out baseline split' },
    baseline: demoBaselineReport,
    treatmentStatus: 'pending',
    metrics: [
      { key: 'accuracy', label: 'Accuracy', baseline: 0.7852, treatment: null },
      { key: 'balancedAccuracy', label: 'Balanced Accuracy', baseline: 0.7044, treatment: null },
      { key: 'macroF1', label: 'Macro F1', baseline: 0.7195, treatment: null },
      { key: 'rocAuc', label: 'ROC-AUC', baseline: 0.8301, treatment: null },
      { key: 'mcc', label: 'MCC', baseline: 0.4547, treatment: null },
      { key: 'brier', label: 'Brier Score', baseline: 0.1483, treatment: null, lowerIsBetter: true },
    ],
  };
}
