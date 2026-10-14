import React, { useEffect, useState } from 'react';
import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Link } from 'react-router-dom';
import { getAuditOverview } from '../api.js';
import SampleDataBanner from '../components/audit/SampleDataBanner.jsx';

const format = (value) => new Intl.NumberFormat('en-US').format(value);
const Card = ({ title, children, caption }) => <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6"><h2 className="font-bold tracking-tight">{title}</h2><div className="mt-5">{children}</div><p className="mt-4 text-sm leading-6 text-slate-500">{caption}</p></section>;

export default function LabelAuditOverview() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const load = () => { setError(false); getAuditOverview().then(setData).catch(() => setError(true)); };
  useEffect(load, []);
  if (error) return <Retry onClick={load} />;
  if (!data) return <Loading />;
  const agreement = [{ name: 'All 3 agree', value: data.agentAgreement.allAgree, fill: '#4f46e5' }, { name: '2 of 3 agree', value: data.agentAgreement.twoOfThree, fill: '#8b5cf6' }, { name: 'No majority', value: data.agentAgreement.noMajority, fill: '#cbd5e1' }];
  return (
    <>
      <Header title="Multi-agent label audit workflow" subtitle="The finished product audits selected issue labels with Policy, Data, and Pattern agents, then preserves a Judge decision and full evidence trail." />
      <SampleDataBanner />
      <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><Stat label="Dataset issues" value={format(data.totalIssues)} /><Stat label="Planned audit sample" value={format(data.auditSampleSize)} detail="Representative subset" /><Stat label="Labels changed" value={format(data.labelsChanged)} /><Stat label="Audit status" value={data.status === 'planned' ? 'Pending' : 'Active'} /></div>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Audit funnel" caption="The full population remains intact; expensive agent reasoning is applied to a selected subset."><div className="space-y-4">{[{ label: 'Canonical dataset', count: data.totalIssues, width: '100%' }, { label: 'Selective audit sample', count: data.auditSampleSize, width: '62%' }, { label: 'Completed audited records', count: data.labelsChanged ? data.auditSampleSize : 0, width: data.labelsChanged ? '62%' : '8%' }].map((item) => <div key={item.label}><div className="mb-1 flex justify-between text-sm"><span>{item.label}</span><b>{format(item.count)}</b></div><div className="h-9 rounded-r-lg bg-indigo-500" style={{ width: item.width }} /></div>)}</div></Card>
        <Card title="Agent agreement" caption="Agreement and disagreement become research signals after the audit run."><div className="flex h-64 items-center"><ResponsiveContainer width="58%" height="100%"><PieChart><Pie data={agreement} dataKey="value" innerRadius={58} outerRadius={90} paddingAngle={3}>{agreement.map((a) => <Cell key={a.name} fill={a.fill} />)}</Pie><Tooltip formatter={format} /></PieChart></ResponsiveContainer><div className="space-y-3 text-sm">{agreement.map((a) => <div key={a.name} className="flex gap-2"><i className="mt-1.5 h-3 w-3 rounded-full" style={{ background: a.fill }} /><span>{a.name}<b className="block">{format(a.value)}</b></span></div>)}</div></div></Card>
        <Card title="Change rate by source" caption="This chart will populate after real audited records exist."><ResponsiveContainer width="100%" height={240}><BarChart data={data.bySource}><XAxis dataKey="source" axisLine={false} tickLine={false} /><YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} axisLine={false} tickLine={false} /><Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} /><Bar dataKey="changeRate" fill="#4f46e5" radius={[7, 7, 0, 0]} /></BarChart></ResponsiveContainer></Card>
        <Card title="Agent responsibilities" caption="Responsibilities stay separated so disagreement can be studied."><div className="grid gap-3 text-sm leading-6 text-slate-700"><p><b>Policy Agent:</b> checks explicit defect policy.</p><p><b>Data Agent:</b> inspects structured metadata consistency.</p><p><b>Pattern Agent:</b> reviews linguistic and historical defect patterns.</p><p><b>Judge:</b> consolidates evidence without overwriting original_label.</p></div></Card>
      </div>
      <Link to="/audit/cases" className="mt-6 inline-flex rounded-xl bg-indigo-600 px-5 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-200 hover:bg-indigo-700">Open audit case workspace</Link>
    </>
  );
}

function Header({ title, subtitle }) { return <div className="max-w-3xl"><p className="page-kicker text-xs font-bold uppercase tracking-[.18em] text-indigo-600">Audit layer</p><h1 className="page-title mt-3 text-3xl font-extrabold text-slate-950 sm:text-4xl">{title}</h1><p className="mt-3 text-base leading-7 text-slate-600">{subtitle}</p></div>; }
function Stat({ label, value, detail }) { return <article className="stat-card rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-xs font-bold uppercase tracking-[.14em] text-slate-400">{label}</p><p className="stat-value mt-3 text-3xl font-extrabold">{value}</p>{detail && <p className="mt-1 text-sm text-slate-500">{detail}</p>}</article>; }
function Loading() { return <div className="grid min-h-[460px] place-items-center"><span className="h-9 w-9 animate-spin rounded-full border-4 border-indigo-100 border-t-indigo-600" /></div>; }
function Retry({ onClick }) { return <div className="rounded-2xl border border-rose-100 bg-white p-8 text-center"><p>Audit data could not load.</p><button onClick={onClick} className="mt-4 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white">Try again</button></div>; }
