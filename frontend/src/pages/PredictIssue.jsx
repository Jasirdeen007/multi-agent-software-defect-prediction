import React, { useState } from 'react';
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { predictIssue } from '../api.js';

function PredictIssue() {
  const [form, setForm] = useState({ source: 'GitHub', project: '', title: '', description: '', component: '', priority: '' });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const update = (event) => setForm((previous) => ({ ...previous, [event.target.name]: event.target.value }));
  const submit = async (event) => { event.preventDefault(); setLoading(true); setResult(null); try { setResult(await predictIssue(form)); } finally { setLoading(false); } };
  return (
    <>
      <div className="max-w-3xl"><p className="page-kicker text-xs font-bold uppercase tracking-[0.18em] text-indigo-600">Inference layer</p><h1 className="page-title mt-3 text-3xl font-extrabold text-slate-950 sm:text-4xl">Predict defect likelihood for a new issue</h1><p className="mt-3 text-base leading-7 text-slate-600">The production path is model_text to DeBERTa-v3 embedding to XGBoost prediction, followed by confidence-based human review routing.</p></div>
      <section className="mt-8 grid gap-6 lg:grid-cols-2 lg:items-stretch">
        <form onSubmit={submit} className="content-card rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
          <h2 className="text-lg font-bold tracking-tight">Issue report</h2>
          <p className="mt-1 text-sm text-slate-500">Only title and description are used by the current baseline model. Metadata is kept for context and future review logging.</p>
          <div className="mt-6 grid gap-4 sm:grid-cols-2"><Field label="Source" name="source" value={form.source} onChange={update} /><Field label="Project" name="project" value={form.project} onChange={update} placeholder="vscode" /></div>
          <Field label="Title" name="title" value={form.title} onChange={update} placeholder="App crashes when applying a coupon" required />
          <label className="mt-5 block text-sm font-semibold text-slate-700">Description<textarea required name="description" value={form.description} onChange={update} placeholder="Describe observed behavior, expected behavior, and reproduction steps." className="form-field mt-2 block h-44 w-full resize-none rounded-xl border border-slate-300 px-4 py-3 text-sm leading-6 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100" /></label>
          <button disabled={loading} className="primary-action mt-6 inline-flex w-full items-center justify-center rounded-xl bg-indigo-600 px-5 py-3.5 text-sm font-bold text-white transition hover:bg-indigo-700 disabled:cursor-wait disabled:opacity-70">{loading ? 'Running inference...' : 'Run Prediction'}</button>
        </form>
        <ResultPanel result={result} loading={loading} />
      </section>
    </>
  );
}

function Field({ label, required, ...props }) {
  return <label className="mt-5 block text-sm font-semibold text-slate-700">{label}<input required={required} {...props} className="form-field mt-2 block w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100" /></label>;
}

function ResultPanel({ result, loading }) {
  if (loading) return <section className="grid min-h-[520px] place-items-center rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><div className="text-center"><span className="mx-auto block h-9 w-9 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /><p className="mt-4 text-sm font-semibold text-slate-600">Generating prediction...</p></div></section>;
  if (!result) return <section className="result-empty grid min-h-[520px] place-items-center rounded-2xl border border-dashed border-slate-300 p-6 text-center"><div><span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-indigo-50 text-xl text-indigo-600">AI</span><h2 className="mt-4 text-lg font-bold">Prediction output appears here</h2><p className="mx-auto mt-2 max-w-xs text-sm leading-6 text-slate-500">Submit an issue to view model label, confidence, explanation signals, and review routing.</p></div></section>;
  const bug = result.label === 'bug'; const chartData = result.topFeatures.map((item) => ({ ...item, weight: Math.round(item.weight * 100) }));
  return <section className="content-card rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7"><p className="text-xs font-bold uppercase tracking-[0.15em] text-slate-400">Prediction result</p><div className="mt-4 flex flex-wrap items-center justify-between gap-4"><div><span className={`inline-flex rounded-full px-3 py-1.5 text-sm font-bold ${bug ? 'bg-rose-50 text-rose-700' : 'bg-emerald-50 text-emerald-700'}`}>{bug ? 'Bug' : 'Non-bug'}</span><h2 className="mt-3 text-xl font-bold tracking-tight">{bug ? 'Likely software defect' : 'Likely non-defect issue'}</h2><p className="mt-1 text-sm text-slate-500">{result.modelVersion}</p></div><div className="text-right"><b className="stat-value text-2xl font-extrabold">{result.confidence}%</b><span className="block text-xs font-medium text-slate-500">confidence</span></div></div><div className="mt-5 h-2.5 overflow-hidden rounded-full bg-slate-100"><div className={bug ? 'h-full rounded-full bg-rose-500' : 'h-full rounded-full bg-emerald-500'} style={{ width: `${result.confidence}%` }} /></div>{result.reviewRequired && <p className="mt-5 rounded-xl bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-800">Low or moderate confidence: route this case to a human tester review queue.</p>}<p className="mt-5 text-sm leading-6 text-slate-600">{result.summary}</p><div className="mt-7 border-t border-slate-100 pt-6"><div className="flex items-baseline justify-between gap-3"><h3 className="font-bold">Explanation signals</h3><span className="text-xs text-slate-400">Embedding-level preview</span></div><ResponsiveContainer width="100%" height={210}><BarChart data={chartData} layout="vertical" margin={{ top: 20, right: 10, left: 12, bottom: 0 }}><XAxis type="number" hide domain={[0, 45]} /><YAxis type="category" dataKey="feature" width={125} axisLine={false} tickLine={false} tick={{ fill: '#475569', fontSize: 12 }} /><Tooltip formatter={(value) => `${value}% influence`} cursor={{ fill: '#f8fafc' }} /><Bar dataKey="weight" radius={[0, 6, 6, 0]}>{chartData.map((item) => <Cell key={item.feature} fill={item.direction === 'bug' ? '#f43f5e' : '#10b981'} />)}</Bar></BarChart></ResponsiveContainer></div></section>;
}

export default PredictIssue;
