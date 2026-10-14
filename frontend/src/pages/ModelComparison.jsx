import React, { useEffect, useState } from 'react';
import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getModelComparison } from '../api.js';
import MetricsTable from '../components/audit/MetricsTable.jsx';

export default function ModelComparison() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const load = () => { setError(false); getModelComparison().then(setData).catch(() => setError(true)); };
  useEffect(load, []);
  if (error) return <div className="text-center"><p>Model data could not load.</p><button onClick={load} className="mt-3 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white">Try again</button></div>;
  if (!data) return <div className="grid min-h-[460px] place-items-center"><span className="h-9 w-9 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /></div>;
  const chartMetrics = data.metrics.filter((m) => ['accuracy', 'balancedAccuracy', 'macroF1', 'rocAuc'].includes(m.key)).map((m) => ({ name: m.label, Baseline: m.baseline, Treatment: m.treatment ?? 0 }));
  const split = data.baseline.split;
  return (
    <>
      <div className="max-w-3xl"><p className="page-kicker text-xs font-bold uppercase tracking-[.18em] text-indigo-600">Model layer</p><h1 className="page-title mt-3 text-3xl font-extrabold text-slate-950 sm:text-4xl">Original-label baseline report</h1><p className="mt-3 text-base leading-7 text-slate-600">The baseline is complete. The audited-label treatment model will appear here after KG-guided multi-agent auditing creates the treatment labels.</p></div>
      <section className="mt-6 grid gap-px overflow-hidden rounded-2xl border border-slate-200 bg-slate-200 sm:grid-cols-3"><Setup label="Baseline labels" value={data.setup.baselineLabelSource} /><Setup label="Treatment labels" value={data.setup.treatmentLabelSource} /><Setup label="Evaluation split" value={data.setup.testSet} /></section>
      <section className="mt-6 grid gap-4 sm:grid-cols-4"><Stat label="Train" value={split.train.toLocaleString()} /><Stat label="Validation" value={split.validation.toLocaleString()} /><Stat label="Test" value={split.test.toLocaleString()} /><Stat label="Threshold" value={data.baseline.threshold.toFixed(2)} /></section>
      <section className="mt-6"><h2 className="mb-3 text-lg font-bold">Metrics</h2><MetricsTable metrics={data.metrics} /></section>
      <section className="mt-6 grid gap-6 lg:grid-cols-2"><Chart title="Baseline metrics" caption="Treatment bars are intentionally empty until audited-label training is complete."><ResponsiveContainer width="100%" height={280}><BarChart data={chartMetrics} margin={{ left: -18 }}><XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} /><YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} axisLine={false} tickLine={false} /><Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} /><Legend /><Bar dataKey="Baseline" fill="#4f46e5" radius={[5, 5, 0, 0]} /><Bar dataKey="Treatment" fill="#cbd5e1" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></Chart><Chart title="Model input contract" caption="This keeps baseline and treatment comparable."><div className="grid gap-3 text-sm leading-6 text-slate-700"><p><b>Input feature:</b> {data.baseline.inputFeatures.join(', ')}</p><p><b>Target:</b> {data.baseline.target}</p><p><b>Tracking fields:</b> {data.baseline.trackingFields.join(', ')}</p><p><b>Architecture:</b> {data.baseline.model}</p><p><b>Status:</b> baseline complete, treatment pending.</p></div></Chart></section>
      <section className="mt-6 rounded-2xl border border-slate-200 bg-slate-100 p-5 text-sm leading-6 text-slate-600"><b className="text-slate-800">Research boundary.</b> Do not claim audited-label improvement until the treatment dataset is created, trained, and evaluated with the same split and protocol.</section>
    </>
  );
}

function Setup({ label, value }) { return <div className="bg-white p-5"><p className="text-xs font-bold uppercase tracking-[.14em] text-slate-400">{label}</p><p className="mt-2 text-sm font-bold text-slate-800">{value}</p></div>; }
function Stat({ label, value }) { return <article className="stat-card rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-xs font-bold uppercase tracking-[.14em] text-slate-400">{label}</p><p className="stat-value mt-3 text-3xl font-extrabold">{value}</p></article>; }
function Chart({ title, children, caption }) { return <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6"><h2 className="font-bold">{title}</h2><div className="mt-5">{children}</div><p className="mt-4 text-sm leading-6 text-slate-500">{caption}</p></section>; }
