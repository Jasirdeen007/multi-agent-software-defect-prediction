export const auditOverviewMock = {
  totalIssues: 926461, auditSampleSize: 5000, labelsChanged: 640, labelChangeRate: 0.128,
  agentAgreement: { allAgree: 3100, twoOfThree: 1450, noMajority: 450 },
  changeDirection: [{ from: 'Bug', to: 'Not a Bug', count: 390 }, { from: 'Not a Bug', to: 'Bug', count: 250 }],
  bySource: [{ source: 'Bugzilla', audited: 2000, changed: 210, changeRate: 0.105 }, { source: 'GitHub', audited: 1750, changed: 300, changeRate: 0.171 }, { source: 'JIRA', audited: 1250, changed: 130, changeRate: 0.104 }],
};

const titles = [
  ['GitHub', 'checkout-ui', '1432', 'App crashes when saving an empty form', 'Non-Bug', 'Bug', true, .82, '2 of 3'],
  ['Bugzilla', 'firefox', '1902341', 'Tabs stop responding after restoring a session', 'Bug', 'Bug', false, .91, '3 of 3'],
  ['JIRA', 'kafka', 'KAFKA-18342', 'Add a shortcut for pausing consumers', 'Non-Bug', 'Non-Bug', false, .76, '3 of 3'],
  ['GitHub', 'vscode', '198765', 'Search results disappear after changing folders', 'Bug', 'Bug', false, .88, '3 of 3'],
  ['Bugzilla', 'thunderbird', '1898762', 'Update the account settings wording', 'Bug', 'Non-Bug', true, .71, '2 of 3'],
  ['JIRA', 'spark', 'SPARK-48710', 'Job fails when reading a partitioned parquet file', 'Non-Bug', 'Bug', true, .84, '2 of 3'],
  ['GitHub', 'kubernetes', '128901', 'Document expected retry behaviour', 'Non-Bug', 'Non-Bug', false, .79, '3 of 3'],
  ['Bugzilla', 'eclipse', '592014', 'Editor freezes after closing a large project', 'Bug', 'Bug', false, .89, '3 of 3'],
  ['JIRA', 'jenkins', 'JENKINS-74511', 'Pipeline stage reports success after timeout', 'Bug', 'Bug', false, .86, '2 of 3'],
  ['GitHub', 'chromium', '152344', 'Button label is confusing on mobile', 'Bug', 'Non-Bug', true, .67, 'No majority'],
  ['Bugzilla', 'firefox', '1901043', 'Unable to open a downloaded PDF file', 'Non-Bug', 'Bug', true, .81, '2 of 3'],
  ['JIRA', 'hadoop', 'HADOOP-19220', 'Improve text in the storage warning', 'Non-Bug', 'Non-Bug', false, .74, '3 of 3'],
  ['GitHub', 'react', '30215', 'Build fails with an undefined module error', 'Bug', 'Bug', false, .92, '3 of 3'],
  ['Bugzilla', 'libreoffice', '160211', 'New icon suggestion for export action', 'Bug', 'Non-Bug', true, .69, '2 of 3'],
  ['JIRA', 'kafka', 'KAFKA-18356', 'Consumer offset is committed twice', 'Bug', 'Bug', false, .90, '3 of 3'],
  ['GitHub', 'vscode', '198902', 'Support a custom indent guide colour', 'Non-Bug', 'Non-Bug', false, .78, '3 of 3'],
  ['Bugzilla', 'thunderbird', '1899320', 'Message window closes unexpectedly', 'Non-Bug', 'Bug', true, .83, '2 of 3'],
  ['JIRA', 'spark', 'SPARK-48722', 'Clarify the streaming configuration example', 'Bug', 'Non-Bug', true, .66, 'No majority'],
  ['GitHub', 'kubernetes', '129044', 'Deployment ignores the configured timeout', 'Bug', 'Bug', false, .87, '3 of 3'],
  ['Bugzilla', 'eclipse', '592127', 'Add a preference for compact toolbars', 'Non-Bug', 'Non-Bug', false, .75, '3 of 3'],
  ['JIRA', 'jenkins', 'JENKINS-74526', 'Agent connection is lost during checkout', 'Bug', 'Bug', false, .88, '2 of 3'],
  ['GitHub', 'chromium', '152501', 'Screen reader focus skips the dialog heading', 'Non-Bug', 'Bug', true, .77, '2 of 3'],
  ['Bugzilla', 'firefox', '1902910', 'Suggested preference name is too technical', 'Bug', 'Non-Bug', true, .68, 'No majority'],
  ['JIRA', 'hadoop', 'HADOOP-19244', 'Node manager fails after a disk fills up', 'Non-Bug', 'Bug', true, .85, '2 of 3'],
  ['GitHub', 'react', '30302', 'Improve warning shown for duplicate keys', 'Non-Bug', 'Non-Bug', false, .73, '3 of 3'],
];

const baseAuditCases = titles.map(([source, project, issueId, title, originalLabel, auditedLabel, changed, judgeConfidence, agentAgreement], index) => ({ source, project, issueId, title, originalLabel, auditedLabel, changed, judgeConfidence, agentAgreement, index }));

// Eight complete pages make the explorer controls meaningful in preview mode.
export const auditCasesMock = Array.from({ length: 80 }, (_, index) => {
  const base = baseAuditCases[index % baseAuditCases.length];
  const cycle = Math.floor(index / baseAuditCases.length);
  return {
    ...base,
    id: `case_${String(index + 1).padStart(4, '0')}`,
    issueId: cycle ? `${base.issueId}-${cycle + 1}` : base.issueId,
    title: cycle ? `${base.title} — follow-up ${cycle + 1}` : base.title,
  };
});

export const caseDetails = Object.fromEntries(auditCasesMock.map((item) => [item.id, {
  ...item,
  description: `This illustrative report concerns ${item.title.toLowerCase()}. The reporter supplied observed behaviour, expected behaviour, and steps that the audit team reviewed.`,
  agents: [
    { name: 'Policy Agent', verdict: item.auditedLabel, confidence: Math.min(.95, item.judgeConfidence + .03), reason: 'Checks whether the report describes unexpected behaviour against the project policy.' },
    { name: 'Data Agent', verdict: item.changed ? item.originalLabel : item.auditedLabel, confidence: Math.max(.55, item.judgeConfidence - .04), reason: 'Reviews the available issue fields and their consistency with the report.' },
    { name: 'Pattern Agent', verdict: item.auditedLabel, confidence: Math.max(.55, item.judgeConfidence - .08), reason: 'Compares the wording with similar issues from the same project.' },
  ],
  knowledgeGraphEvidence: ['Similar issues in the same project were reviewed as part of the audit.', `Project context for ${item.project} was included as supporting evidence.`],
  judge: { finalLabel: item.auditedLabel, confidence: item.judgeConfidence, rationale: `${item.agentAgreement} agents informed the final decision; the Judge preserved their individual evidence.` },
}]));

export const modelComparisonMock = {
  metrics: [
    { key: 'accuracy', label: 'Accuracy', baseline: .87, treatment: .88 }, { key: 'balancedAccuracy', label: 'Balanced Accuracy', baseline: .83, treatment: .85 },
    { key: 'precision', label: 'Precision', baseline: .89, treatment: .90 }, { key: 'recall', label: 'Recall', baseline: .90, treatment: .91 },
    { key: 'f1', label: 'F1 Score', baseline: .89, treatment: .90 }, { key: 'rocAuc', label: 'ROC-AUC', baseline: .93, treatment: .94 },
    { key: 'mcc', label: 'MCC', baseline: .68, treatment: .71 }, { key: 'brier', label: 'Brier Score', baseline: .095, treatment: .088, lowerIsBetter: true },
  ],
  calibration: { baseline: [{ predicted: .1, actual: .08 }, { predicted: .3, actual: .27 }, { predicted: .5, actual: .48 }, { predicted: .7, actual: .66 }, { predicted: .9, actual: .84 }], treatment: [{ predicted: .1, actual: .09 }, { predicted: .3, actual: .29 }, { predicted: .5, actual: .51 }, { predicted: .7, actual: .69 }, { predicted: .9, actual: .88 }] },
  setup: { baselineLabelSource: 'Original BugHub labels', treatmentLabelSource: 'Agent-audited labels', testSet: 'Held-out test split' },
};
