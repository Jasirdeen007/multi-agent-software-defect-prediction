import React, { useEffect, useState } from 'react';
import { Bar, BarChart, Cell, LabelList, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getDatasetStats } from '../api.js';

const classColors = ['#4f46e5', '#cbd5e1'];
const sourceColors = ['#6366f1', '#8b5cf6', '#14b8a6'];
const formatNumber = (value) => new Intl.NumberFormat('en-US').format(value);

function DatasetInsights() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');
  const load = () => { setError(''); getDatasetStats().then(setStats).catch((err) => setError(err.message || 'Dataset summary could not be loaded.')); };
  useEffect(load, []);
  if (error) return <div role="alert" className="rounded-2xl border border-rose-200 bg-white p-8"><h1 className="text-xl font-bold">Dataset summary unavailable</h1><p className="mt-2 text-sm text-slate-600">{error}</p><button onClick={load} className="mt-4 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white">Retry</button></div>;
  if (!stats) return <Loading />;
  return (
    <>
      <div className="max-w-3xl">
        <p className="page-kicker text-xs font-bold uppercase tracking-[0.18em] text-indigo-600">Data layer</p>
        <h1 className="page-title mt-3 text-3xl font-extrabold text-slate-950 sm:text-4xl">Canonical BugHub issue population</h1>
        <p className="mt-3 text-base leading-7 text-slate-600">The dashboard starts from the validated canonical Parquet dataset and preserves the original BugHub source labels for baseline comparison.</p>
      </div>
      <section className="mt-8 grid gap-4 sm:grid-cols-4">
        <Stat label="Canonical records" value={formatNumber(stats.totalIssues)} detail="Stage 1 artifact" />
        <Stat label="Baseline rows" value={formatNumber(stats.baselineRows)} detail={`${stats.removedShortTextRows} short-text rows removed`} />
        <Stat label="Projects" value={stats.totalProjects} detail="Composite identity preserved" />
        <Stat label="Sources" value="3" detail="Bugzilla, GitHub, JIRA" />
      </section>
      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <ChartCard title="Original label distribution" caption="These are BugHub source labels, not human gold labels.">
          <div className="flex h-64 items-center justify-center gap-4">
            <ResponsiveContainer width="54%" height="100%"><PieChart><Pie data={stats.classDistribution} dataKey="count" nameKey="label" innerRadius={62} outerRadius={91} paddingAngle={3}>{stats.classDistribution.map((entry, index) => <Cell key={entry.label} fill={classColors[index]} />)}</Pie><Tooltip formatter={(value) => formatNumber(value)} /></PieChart></ResponsiveContainer>
            <div className="space-y-4">{stats.classDistribution.map((item, index) => <div key={item.label} className="flex items-start gap-3"><i className="mt-1.5 h-3 w-3 rounded-full" style={{ backgroundColor: classColors[index] }} /><div><b className="block capitalize text-slate-800">{item.label}</b><span className="text-sm text-slate-500">{item.percent}% / {formatNumber(item.count)}</span></div></div>)}</div>
          </div>
        </ChartCard>
        <ChartCard title="Issues by tracker" caption="Source-specific fields remain missing where the tracker does not provide them.">
          <ResponsiveContainer width="100%" height={256}><BarChart data={stats.sources} margin={{ top: 12, right: 8, left: -18, bottom: 4 }}><XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 12 }} /><YAxis axisLine={false} tickLine={false} tickFormatter={(value) => `${Math.round(value / 1000)}k`} tick={{ fill: '#94a3b8', fontSize: 11 }} /><Tooltip formatter={(value) => formatNumber(value)} cursor={{ fill: '#f8fafc' }} /><Bar dataKey="count" radius={[8, 8, 0, 0]}>{stats.sources.map((source, index) => <Cell key={source.name} fill={sourceColors[index]} />)}</Bar></BarChart></ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Field availability" caption="The baseline model uses model_text only; metadata supports later KG and audit context.">
          <div className="overflow-x-auto"><table className="w-full min-w-[390px] text-left text-sm"><thead><tr className="border-b border-slate-100 text-xs uppercase tracking-wider text-slate-400"><th className="pb-3 font-semibold">Source</th>{['Severity', 'Priority', 'Component', 'Resolution'].map((field) => <th key={field} className="pb-3 text-center font-semibold">{field}</th>)}</tr></thead><tbody>{stats.completeness.map((row) => <tr key={row.source} className="border-b border-slate-100 last:border-0"><td className="py-4 font-semibold text-slate-700">{row.source}</td>{['severity', 'priority', 'component', 'resolution'].map((field) => <td key={field} className="py-4 text-center"><span className={`inline-grid h-6 w-6 place-items-center rounded-full text-xs font-bold ${row[field] ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400'}`}>{row[field] ? 'Yes' : '-'}</span></td>)}</tr>)}</tbody></table></div>
        </ChartCard>
        <ChartCard title="Top projects" caption="Project concentration matters for split design, leakage checks, and later audit sampling.">
          <ResponsiveContainer width="100%" height={300}><BarChart data={stats.topProjects} layout="vertical" margin={{ top: 0, right: 32, left: 45, bottom: 0 }}><XAxis type="number" hide /><YAxis type="category" dataKey="project" axisLine={false} tickLine={false} width={125} tick={{ fill: '#475569', fontSize: 11 }} /><Tooltip formatter={(value) => formatNumber(value)} cursor={{ fill: '#f8fafc' }} /><Bar dataKey="count" fill="#4f46e5" radius={[0, 6, 6, 0]} barSize={15}><LabelList dataKey="count" position="right" formatter={formatNumber} style={{ fill: '#64748b', fontSize: 10 }} /></Bar></BarChart></ResponsiveContainer>
        </ChartCard>
      </section>
    </>
  );
}

function ChartCard({ title, children, caption }) { return <section className="chart-card rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6"><h2 className="text-base font-bold tracking-tight text-slate-900">{title}</h2><div className="mt-5">{children}</div><p className="mt-4 text-sm leading-6 text-slate-500">{caption}</p></section>; }
function Stat({ label, value, detail }) { return <article className="stat-card rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-xs font-bold uppercase tracking-[0.15em] text-slate-400">{label}</p><p className="stat-value mt-3 text-3xl font-extrabold text-slate-900">{value}</p><p className="mt-1 text-sm text-slate-500">{detail}</p></article>; }
function Loading() { return <div className="grid min-h-[460px] place-items-center rounded-2xl border border-slate-200 bg-white"><span className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /></div>; }
export default DatasetInsights;
