import React, { useState } from 'react';
import { predictIssue } from '../api.js';

export default function PredictIssue() {
  const [form, setForm] = useState({ source: 'GitHub', project: '', title: '', description: '', component: '', priority: '' });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const update = (event) => setForm((previous) => ({ ...previous, [event.target.name]: event.target.value }));
  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setResult(null);
    setError('');
    try {
      setResult(await predictIssue(form));
    } catch (requestError) {
      setError(requestError.message || 'Treatment inference is unavailable.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="max-w-3xl">
        <p className="page-kicker text-xs font-bold uppercase tracking-[0.18em] text-indigo-600">Treatment inference</p>
        <h1 className="page-title mt-3 text-3xl font-extrabold text-slate-950 sm:text-4xl">Analyze a software issue</h1>
        <p className="mt-3 text-base leading-7 text-slate-600">User-facing inference is reserved for a treatment model with verified audit provenance and evaluation. The input contract is title plus description, embedded with DeBERTa-v3 and classified by XGBoost.</p>
      </div>

      <section className="mt-8 grid gap-6 lg:grid-cols-2 lg:items-stretch">
        <form onSubmit={submit} className="content-card rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
          <h2 className="text-lg font-bold tracking-tight">Issue report</h2>
          <p className="mt-1 text-sm text-slate-500">Only title and description enter the model. Source and project are contextual metadata.</p>
          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <Field label="Source" name="source" value={form.source} onChange={update} />
            <Field label="Project" name="project" value={form.project} onChange={update} placeholder="vscode" />
          </div>
          <Field label="Title" name="title" value={form.title} onChange={update} placeholder="App crashes when applying a coupon" required />
          <label className="mt-5 block text-sm font-semibold text-slate-700">
            Description
            <textarea required name="description" value={form.description} onChange={update} placeholder="Describe observed behavior, expected behavior, and reproduction steps." className="form-field mt-2 block h-44 w-full resize-none rounded-xl border border-slate-300 px-4 py-3 text-sm leading-6 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100" />
          </label>
          {error && <p role="alert" className="mt-5 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm leading-6 text-rose-800">{error}</p>}
          <button disabled={loading} className="primary-action mt-6 inline-flex w-full items-center justify-center rounded-xl bg-indigo-600 px-5 py-3.5 text-sm font-bold text-white transition hover:bg-indigo-700 disabled:cursor-wait disabled:opacity-70">
            {loading ? 'Running treatment model...' : 'Run treatment prediction'}
          </button>
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
  if (loading) return <section className="grid min-h-[520px] place-items-center rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><div className="text-center"><span className="mx-auto block h-9 w-9 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /><p className="mt-4 text-sm font-semibold text-slate-600">Generating treatment prediction...</p></div></section>;
  if (!result) return <section className="result-empty grid min-h-[520px] place-items-center rounded-2xl border border-dashed border-slate-300 p-6 text-center"><div><span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-indigo-50 text-xl text-indigo-600">ML</span><h2 className="mt-4 text-lg font-bold">Treatment result</h2><p className="mx-auto mt-2 max-w-xs text-sm leading-6 text-slate-500">Run a report through the treatment model to see its Bug probability and classification threshold.</p></div></section>;

  const isBug = result.label === 'bug';
  const probabilityBug = Math.round(result.probabilityBug * 100);
  return (
    <section className="content-card rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
      <p className="text-xs font-bold uppercase tracking-[0.15em] text-slate-400">Treatment prediction</p>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className={`inline-flex rounded-full px-3 py-1.5 text-sm font-bold ${isBug ? 'bg-rose-50 text-rose-700' : 'bg-emerald-50 text-emerald-700'}`}>{isBug ? 'Bug' : 'Non-bug'}</span>
          <h2 className="mt-3 text-xl font-bold tracking-tight">{isBug ? 'Classified as Bug' : 'Classified as Non-bug'}</h2>
          <p className="mt-1 text-sm text-slate-500">{result.modelName} · {result.modelVersion}</p>
        </div>
        <div className="text-right"><b className="stat-value text-2xl font-extrabold">{probabilityBug}%</b><span className="block text-xs font-medium text-slate-500">P(Bug)</span></div>
      </div>
      <div className="mt-5 h-2.5 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-rose-500" style={{ width: `${probabilityBug}%` }} /></div>
      <div className="mt-5 grid gap-3 rounded-xl bg-slate-50 p-4 text-sm sm:grid-cols-2">
        <p><span className="text-slate-500">Classification threshold</span><b className="ml-2 text-slate-800">{result.classificationThreshold.toFixed(2)}</b></p>
        <p><span className="text-slate-500">Request</span><b className="ml-2 font-mono text-xs text-slate-800">{result.predictionId}</b></p>
      </div>
      <p className="mt-5 text-sm leading-6 text-slate-600">The displayed probability is the model output; it is not presented as calibrated confidence.</p>
      <div className="mt-6 border-t border-slate-100 pt-5"><h3 className="font-bold">Review signals</h3><p className="mt-2 text-sm leading-6 text-slate-600">Low-confidence review: {result.lowConfidenceCheck}. OOD detection: {result.oodCheck}.</p></div>
    </section>
  );
}