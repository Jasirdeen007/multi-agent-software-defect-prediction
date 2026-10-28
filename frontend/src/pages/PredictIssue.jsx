import React, { useRef, useState } from 'react';
import { predictIssue, predictIssueFile } from '../api.js';

export default function PredictIssue() {
  const [form, setForm] = useState({ source: 'GitHub', project: '', title: '', description: '', component: '', priority: '' });
  const [result, setResult] = useState(null);
  const [mode, setMode] = useState('single');
  const [file, setFile] = useState(null);
  const [batchResult, setBatchResult] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const update = (event) => setForm((previous) => ({ ...previous, [event.target.name]: event.target.value }));
  const startNewInput = () => {
    setForm({ source: 'GitHub', project: '', title: '', description: '', component: '', priority: '' });
    setResult(null);
    setBatchResult(null);
    setFile(null);
    setError('');
    setUploadError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };
  const changeMode = (nextMode) => {
    startNewInput();
    setMode(nextMode);
  };
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

  const submitFile = async (event) => {
    event.preventDefault();
    if (!file) return;
    setUploading(true);
    setBatchResult(null);
    setUploadError('');
    try {
      setBatchResult(await predictIssueFile(file));
    } catch (requestError) {
      setUploadError(requestError.message || 'File inference is unavailable.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <div className="max-w-3xl">
        <p className="page-kicker text-xs font-bold uppercase tracking-[0.18em] text-indigo-600">Treatment inference</p>
        <h1 className="page-title mt-3 text-3xl font-extrabold text-slate-950 sm:text-4xl">Analyze a software issue</h1>
        <p className="mt-3 text-base leading-7 text-slate-600">Score an issue with the treatment model trained on the completed single-call audited labels. Title and description form the model text; local DeBERTa-v3 embedding and XGBoost prediction run on CUDA.</p>
      </div>

      <div className="mt-8 inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm" role="tablist" aria-label="Prediction input type">
        <button type="button" role="tab" aria-selected={mode === 'single'} onClick={() => changeMode('single')} className={`rounded-lg px-4 py-2.5 text-sm font-semibold ${mode === 'single' ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-50'}`}>Single issue</button>
        <button type="button" role="tab" aria-selected={mode === 'file'} onClick={() => changeMode('file')} className={`rounded-lg px-4 py-2.5 text-sm font-semibold ${mode === 'file' ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-50'}`}>CSV / Excel batch</button>
      </div>

      {mode === 'single' ? <section className="mt-5 grid gap-6 lg:grid-cols-2 lg:items-stretch">
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
            {loading ? 'Running treatment model on GPU...' : 'Run treatment prediction'}
          </button>
        </form>
        <ResultPanel result={result} loading={loading} onNewInput={startNewInput} />
      </section> : <section className="mt-5 grid gap-6 xl:grid-cols-[minmax(320px,.8fr)_minmax(0,1.4fr)]">
        <form onSubmit={submitFile} className="content-card h-fit rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
          <h2 className="text-lg font-bold tracking-tight">Upload issue reports</h2>
          <p className="mt-1 text-sm leading-6 text-slate-500">Choose a CSV or Excel workbook (.xlsx). For workbooks, the first worksheet is used.</p>
          <label className="mt-5 block rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-6 text-center hover:border-indigo-400">
            <span className="block text-sm font-semibold text-slate-800">{file ? file.name : 'Select a .csv or .xlsx file'}</span>
            <span className="mt-1 block text-xs text-slate-500">Up to 15 MB and 1,000 issue rows</span>
            <input ref={fileInputRef} type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" className="sr-only" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setBatchResult(null); setUploadError(''); }} />
          </label>
          <div className="mt-5 rounded-xl bg-indigo-50 p-4 text-sm leading-6 text-indigo-950">
            <b>Accepted columns</b>
            <p className="mt-1">Required: <code>title</code> or precombined <code>model_text</code>. Description may be blank.</p>
            <p className="mt-1">Title aliases: <code>issue_title</code>, <code>issue_summary</code>, <code>summary</code>, <code>subject</code>. Description aliases: <code>issue_description</code>, <code>body</code>, <code>details</code>, <code>text</code>.</p>
            <p className="mt-1">Optional context: <code>source</code>, <code>project</code>, <code>issue_id</code>, <code>component</code>, <code>priority</code>.</p>
          </div>
          <p className="mt-4 text-xs leading-5 text-slate-500">Headers are matched without case sensitivity. Title and description are whitespace-cleaned and combined exactly as model text; optional metadata and any uploaded label columns are not model features.</p>
          {uploadError && <p role="alert" className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{uploadError}</p>}
          <button disabled={!file || uploading} className="primary-action mt-5 inline-flex w-full items-center justify-center rounded-xl bg-indigo-600 px-5 py-3.5 text-sm font-bold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50">{uploading ? 'Scoring rows on GPU...' : 'Run batch predictions'}</button>
          <button type="button" onClick={downloadTemplate} className="mt-3 w-full rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50">Download CSV template</button>
        </form>
        <BatchResultPanel result={batchResult} loading={uploading} onNewInput={startNewInput} />
      </section>}
    </>
  );
}

function Field({ label, required, ...props }) {
  return <label className="mt-5 block text-sm font-semibold text-slate-700">{label}<input required={required} {...props} className="form-field mt-2 block w-full rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100" /></label>;
}

function downloadTemplate() {
  const template = 'source,project,issue_id,title,description\nGitHub,vscode,12345,"Search results disappear after changing folders","Search returns no results after changing workspace folders. Expected matching files to remain visible."\n';
  const url = URL.createObjectURL(new Blob([template], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'issue_prediction_template.csv';
  link.click();
  URL.revokeObjectURL(url);
}

function downloadBatchResults(result) {
  if (!result) return;
  const columns = ['row_number', 'source', 'project', 'issue_id', 'title', 'label', 'probability_bug', 'status', 'error'];
  const escape = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const csv = [columns.join(','), ...result.results.map((row) => columns.map((column) => escape(row[column])).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'issue_prediction_results.csv';
  link.click();
  URL.revokeObjectURL(url);
}

function BatchResultPanel({ result, loading, onNewInput }) {
  if (loading) return <section className="grid min-h-[420px] place-items-center rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><div className="text-center"><span className="mx-auto block h-9 w-9 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /><p className="mt-4 text-sm font-semibold text-slate-600">Preparing inputs and scoring on CUDA...</p><p className="mt-1 text-xs text-slate-400">Rows are embedded in batches; invalid rows are reported separately.</p></div></section>;
  if (!result) return <section className="result-empty grid min-h-[420px] place-items-center rounded-2xl border border-dashed border-slate-300 p-6 text-center"><div><span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-indigo-50 text-xl text-indigo-600">CSV</span><h2 className="mt-4 text-lg font-bold">Batch results</h2><p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-slate-500">Upload an issue file to see a prediction for each valid row. You can download the results after processing.</p></div></section>;

  return <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-wider text-indigo-600">Batch complete</p><h2 className="mt-1 text-lg font-bold">{result.file_name}</h2><p className="mt-1 text-sm text-slate-500">Request {result.request_id} · {result.inference_device?.toUpperCase() || 'No valid rows'}</p></div><div className="flex gap-2"><button type="button" onClick={onNewInput} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">New batch</button><button onClick={() => downloadBatchResults(result)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">Download results CSV</button></div></div>
    <div className="mt-5 grid grid-cols-3 gap-3"><BatchStat label="Rows" value={result.total_rows} /><BatchStat label="Predicted" value={result.completed_rows} /><BatchStat label="Needs attention" value={result.failed_rows} /></div>
    <p className="mt-4 text-xs leading-5 text-slate-500">Model text uses title plus description. Threshold: {Number(result.classification_threshold).toFixed(2)}. Predictions are not calibrated confidence scores.</p>
    <div className="mt-4 max-h-[500px] overflow-auto rounded-xl border border-slate-200"><table className="w-full min-w-[720px] text-left text-sm"><thead className="sticky top-0 bg-slate-50 text-xs uppercase tracking-wide text-slate-500"><tr>{['Row', 'Issue', 'Project', 'Prediction', 'P(Bug)', 'Status'].map((heading) => <th key={heading} className="p-3">{heading}</th>)}</tr></thead><tbody>{result.results.map((row) => <tr key={row.row_number} className="border-t border-slate-100"><td className="p-3 text-slate-500">{row.row_number}</td><td className="max-w-[280px] truncate p-3 font-medium text-slate-800" title={row.title}>{row.title || '—'}{row.error && <span className="mt-1 block whitespace-normal text-xs font-normal text-rose-700">{row.error}</span>}</td><td className="p-3 text-slate-600">{row.project || '—'}</td><td className="p-3">{row.label ? <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${row.label === 'bug' ? 'bg-rose-50 text-rose-700' : 'bg-emerald-50 text-emerald-700'}`}>{row.label}</span> : '—'}</td><td className="p-3 tabular-nums">{typeof row.probability_bug === 'number' ? `${(row.probability_bug * 100).toFixed(1)}%` : '—'}</td><td className="p-3"><span className={row.status === 'completed' ? 'text-emerald-700' : 'text-rose-700'}>{row.status}</span></td></tr>)}</tbody></table></div>
  </section>;
}

function BatchStat({ label, value }) {
  return <div className="rounded-xl bg-slate-50 p-3"><p className="text-[10px] font-bold uppercase tracking-wide text-slate-400">{label}</p><p className="mt-1 text-xl font-extrabold tabular-nums text-slate-900">{Number(value).toLocaleString()}</p></div>;
}

function ResultPanel({ result, loading, onNewInput }) {
  if (loading) return <section className="grid min-h-[520px] place-items-center rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><div className="text-center"><span className="mx-auto block h-9 w-9 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /><p className="mt-4 text-sm font-semibold text-slate-600">Generating treatment prediction on CUDA...</p></div></section>;
  if (!result) return <section className="result-empty grid min-h-[520px] place-items-center rounded-2xl border border-dashed border-slate-300 p-6 text-center"><div><span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-indigo-50 text-xl text-indigo-600">ML</span><h2 className="mt-4 text-lg font-bold">Treatment result</h2><p className="mx-auto mt-2 max-w-xs text-sm leading-6 text-slate-500">Run a report through the audited-label treatment model to see its Bug probability and classification threshold.</p></div></section>;

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
        <p><span className="text-slate-500">Inference device</span><b className="ml-2 uppercase text-slate-800">{result.inferenceDevice || 'Not reported'}</b></p>
        <p><span className="text-slate-500">Request</span><b className="ml-2 font-mono text-xs text-slate-800">{result.predictionId}</b></p>
      </div>
      <p className="mt-5 text-sm leading-6 text-slate-600">The displayed probability is the model output; it is not presented as calibrated confidence.</p>
      <div className="mt-6 border-t border-slate-100 pt-5"><h3 className="font-bold">Review signals</h3><p className="mt-2 text-sm leading-6 text-slate-600">Low-confidence review: {result.lowConfidenceCheck}. OOD detection: {result.oodCheck}.</p></div>
      <button type="button" onClick={onNewInput} className="mt-5 w-full rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm font-bold text-indigo-700 transition hover:bg-indigo-100">New prediction</button>
    </section>
  );
}
