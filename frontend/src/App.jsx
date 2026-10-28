import React from 'react';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import DatasetInsights from './pages/DatasetInsights.jsx';
import PredictIssue from './pages/PredictIssue.jsx';
import LabelAuditOverview from './pages/LabelAuditOverview.jsx';
import AuditCaseExplorer from './pages/AuditCaseExplorer.jsx';
import ModelComparison from './pages/ModelComparison.jsx';

const linkClass = ({ isActive }) => `rounded-lg px-3 py-2 text-sm font-semibold transition ${isActive ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950'}`;

function App() {
  return <div className="dashboard-shell min-h-screen text-slate-900">
    <header className="app-header sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur"><div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-5 py-3 sm:px-8">
      <NavLink to="/" className="flex items-center gap-3" aria-label="AgentTriage home"><span className="brand-mark grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-500 text-lg font-extrabold text-white">A</span><span><b className="block text-base tracking-tight">AgentTriage</b><small className="block text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">Software defect intelligence</small></span></NavLink>
      <nav className="flex max-w-full items-center gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-slate-50 p-1" aria-label="Main navigation"><NavLink end to="/" className={linkClass}>Dataset</NavLink><NavLink to="/predict" className={linkClass}>Inference</NavLink><NavLink to="/audit" className={linkClass}>Agent Audit <span className="ml-1 rounded-full bg-white/20 px-1.5 py-0.5 text-[10px]">1k</span></NavLink><NavLink to="/audit/cases" className={linkClass}>Audit Cases</NavLink><NavLink to="/compare" className={linkClass}>Model comparison</NavLink></nav>
    </div></header>
    <main className="mx-auto max-w-[1440px] px-5 py-8 sm:px-8 sm:py-10"><Routes><Route path="/" element={<DatasetInsights />} /><Route path="/predict" element={<PredictIssue />} /><Route path="/audit" element={<LabelAuditOverview />} /><Route path="/audit/cases" element={<AuditCaseExplorer />} /><Route path="/compare" element={<ModelComparison />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></main>
  </div>;
}
export default App;
