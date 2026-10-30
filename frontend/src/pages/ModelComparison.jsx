import React, { useEffect, useState } from 'react';
import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getModelComparison } from '../api.js';
import MetricsTable from '../components/audit/MetricsTable.jsx';

export default function ModelComparison() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  const load = () => {
    setError('');
    getModelComparison().then(setData).catch((requestError) => setError(requestError.message));
  };
  useEffect(load, []);

  if (error) return <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-800"><p>Model comparison could not load: {error}</p><button onClick={load} className="mt-3 rounded-lg bg-rose-700 px-4 py-2 font-bold text-white">Retry</button></div>;
  if (!data) return <div className="grid min-h-[460px] place-items-center"><span className="h-9 w-9 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /></div>;

  const chartMetrics = data.metrics
    .filter((metric) => ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc'].includes(metric.key))
    .map((metric) => ({ name: metric.label, Baseline: metric.baseline, Treatment: metric.treatment }));
  const split = data.baseline.split;
  const sameTestSplit = Boolean(data.setup.sameTestSplit);
  const configurationUnverified = data.comparisonStatus === 'configuration_unverified';
  const comparable = data.comparisonStatus === 'comparable';
  const protocolBlocked = data.comparisonStatus === 'protocol_blocked';

  return (
    <>
      <div className="max-w-3xl">
        <p className="page-kicker text-xs font-bold uppercase tracking-[.18em] text-indigo-600">Evaluation</p>
        <h1 className="page-title mt-3 text-3xl font-extrabold text-slate-950 sm:text-4xl">Baseline vs treatment</h1>
        <p className="mt-3 text-base leading-7 text-slate-600">Review the retained original-label baseline and the treatment trained with the completed one-call, four-perspective audit. The metrics use the same held-out test identities; matching model configurations have not been verified.</p>
      </div>

      <section className="mt-6 grid gap-px overflow-hidden rounded-xl border border-slate-200 bg-slate-200 sm:grid-cols-3">
        <Setup label="Baseline labels" value={data.setup.baselineLabelSource} />
        <Setup label="Treatment labels" value={data.setup.treatmentLabelSource} />
        <Setup label="Evaluation split" value={data.setup.testSet} />
      </section>

      <section className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Training records" value={formatCount(split.train)} />
        <Stat label="Validation records" value={formatCount(split.validation)} />
        <Stat label="Held-out test records" value={formatCount(split.test)} />
        <Stat label="Test identities" value={sameTestSplit ? 'Matched' : protocolBlocked ? 'Blocked' : 'Unverified'} detail={sameTestSplit ? 'Model configuration match unverified' : protocolBlocked ? 'Historical result shown below' : 'Identity check failed'} />
      </section>

      {!comparable && <p role="status" className="mt-5 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm leading-6 text-amber-950"><b>{configurationUnverified && sameTestSplit ? 'Same test identities; model configuration not verified.' : 'Comparison requires review.'}</b> {configurationUnverified && sameTestSplit ? 'The displayed baseline uses retained regular-baseline metrics, while the tuned baseline is a separate run. Treat metric differences as descriptive; they do not isolate the effect of audited labels.' : protocolBlocked ? data.treatment.reason : 'The baseline and treatment test identities do not match.'} Audit decisions come from one model response and are not independently verified ground truth.</p>}

      <section className="mt-7">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
          <div><h2 className="text-lg font-bold">Held-out test metrics</h2><p className="mt-1 text-sm text-slate-500">Thresholds were selected on validation data, not the held-out test set.</p></div>
          <p className="text-xs font-semibold text-slate-500">Baseline threshold {formatThreshold(data.baseline.threshold)} · Treatment threshold {formatThreshold(data.treatment.threshold)}</p>
        </div>
        <MetricsTable metrics={data.metrics} treatmentStatus={data.treatmentStatus} comparable={comparable} />
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <Chart title="Primary model metrics">
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={chartMetrics} margin={{ left: -18, right: 8 }}>
              <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} axisLine={false} tickLine={false} />
              <Tooltip formatter={(value) => typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'Not recorded'} />
              <Legend />
              <Bar dataKey="Baseline" fill="#526b5d" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Treatment" fill="#d88458" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Chart>
        <Chart title={`Treatment experiment · ${data.treatment.status}`}>
          <dl className="grid gap-4 text-sm sm:grid-cols-2">
            <Detail label="Audited training rows" value={formatCount(data.treatment.recordedAuditedRecords)} />
            <Detail label="Training labels changed" value={formatCount(data.treatment.recordedLabelChanges)} />
            <Detail label="Training records" value={formatCount(data.treatment.trainingRecords)} />
            <Detail label="Treatment threshold" value={formatThreshold(data.treatment.threshold)} />
          </dl>
          <p className="mt-5 border-t border-slate-100 pt-4 text-sm leading-6 text-slate-600">{data.treatment.reason || 'Audit decisions are model-generated, not independently verified ground truth. Validation and test labels remain unchanged.'}</p>
        </Chart>
      </section>

      <section className="mt-6 rounded-xl border border-slate-200 bg-white p-5 text-sm leading-6 text-slate-600">
        <b className="text-slate-900">Interpretation.</b> The treatment summary records Judge-label replacement for audited training rows, with original labels for the rest and unchanged validation/test labels. Since matching XGBoost settings have not been verified, metric differences describe the retained runs and cannot be attributed to label changes alone.
      </section>
    </>
  );
}

function formatCount(value) {
  return Number.isInteger(value) ? value.toLocaleString() : 'Not recorded';
}

function formatThreshold(value) {
  return typeof value === 'number' ? value.toFixed(2) : 'Not recorded';
}

function Setup({ label, value }) {
  return <div className="bg-white p-5"><p className="text-xs font-bold uppercase tracking-[.14em] text-slate-400">{label}</p><p className="mt-2 text-sm font-bold text-slate-800">{value}</p></div>;
}

function Stat({ label, value, detail }) {
  return <article className="stat-card rounded-lg border border-slate-200 bg-white p-5 shadow-sm"><p className="text-xs font-bold uppercase tracking-[.14em] text-slate-400">{label}</p><p className="stat-value mt-3 text-2xl font-extrabold">{value}</p>{detail && <p className="mt-1 text-xs text-slate-500">{detail}</p>}</article>;
}

function Chart({ title, children }) {
  return <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm sm:p-6"><h2 className="font-bold">{title}</h2><div className="mt-5">{children}</div></section>;
}

function Detail({ label, value }) {
  return <div><dt className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</dt><dd className="mt-1 font-bold text-slate-800">{value}</dd></div>;
}
