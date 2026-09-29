import { auditCasesMock, auditOverviewMock, caseDetails, modelComparisonMock } from './mock/auditMock.js';

const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

/** The application's only data boundary; replace these mocks with HTTP later. */
export async function getDatasetStats() {
  await wait(500);
  return {
    totalIssues: 926461,
    totalProjects: 93,
    sources: [
      { name: 'Bugzilla', count: 378941 }, { name: 'GitHub', count: 323870 }, { name: 'JIRA', count: 223650 },
    ],
    classDistribution: [
      { label: 'bug', count: 648841, percent: 70.03 }, { label: 'non-bug', count: 277620, percent: 29.97 },
    ],
    completeness: [
      { source: 'Bugzilla', severity: true, priority: true, component: true, resolution: true },
      { source: 'GitHub', severity: false, priority: false, component: false, resolution: false },
      { source: 'JIRA', severity: true, priority: true, component: true, resolution: true },
    ],
    topProjects: [
      { project: 'Mozilla Firefox', count: 143882 }, { project: 'Visual Studio Code', count: 121405 },
      { project: 'Apache Kafka', count: 98214 }, { project: 'Eclipse Platform', count: 87390 },
      { project: 'Chromium', count: 76840 }, { project: 'Apache Spark', count: 62470 },
      { project: 'Kubernetes', count: 54800 }, { project: 'Jenkins', count: 43990 },
    ],
  };
}

export async function predictIssue({ title, description }) {
  await wait(650);
  const input = `${title} ${description}`.toLowerCase();
  const defectTerms = /crash|error|fail|broken|exception|regression|incorrect|unable|freeze|doesn't work|cannot/.test(input);
  const label = defectTerms ? 'bug' : 'non-bug';
  return {
    label,
    confidence: defectTerms ? 87 : 58,
    topFeatures: defectTerms
      ? [{ feature: 'crashes', weight: 0.38, direction: 'bug' }, { feature: 'when applying', weight: 0.24, direction: 'bug' }, { feature: 'expected behaviour', weight: 0.18, direction: 'bug' }, { feature: 'steps to reproduce', weight: 0.12, direction: 'bug' }]
      : [{ feature: 'would be useful', weight: 0.31, direction: 'non-bug' }, { feature: 'add support', weight: 0.24, direction: 'non-bug' }, { feature: 'new option', weight: 0.17, direction: 'non-bug' }, { feature: 'improve workflow', weight: 0.11, direction: 'non-bug' }],
  };
}

const auditDelay = () => wait(400 + Math.floor(Math.random() * 401));

export async function getAuditOverview() {
  await auditDelay();
  return auditOverviewMock;
}

export async function getAuditCases({ page = 1, pageSize = 10, source = 'all', changedOnly = false, search = '' } = {}) {
  await auditDelay();
  const term = search.trim().toLowerCase();
  const filtered = auditCasesMock.filter((item) => (source === 'all' || item.source === source) && (!changedOnly || item.changed) && (!term || item.title.toLowerCase().includes(term)));
  const safePage = Math.max(1, Math.min(page, Math.max(1, Math.ceil(filtered.length / pageSize))));
  return { total: filtered.length, page: safePage, pageSize, cases: filtered.slice((safePage - 1) * pageSize, safePage * pageSize) };
}

export async function getAuditCaseById(id) {
  await auditDelay();
  return caseDetails[id] ?? null;
}

export async function getModelComparison() {
  await auditDelay();
  return modelComparisonMock;
}
