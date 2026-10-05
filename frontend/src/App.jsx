import React from 'react';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import DatasetInsights from './pages/DatasetInsights.jsx';
import PredictIssue from './pages/PredictIssue.jsx';
import LabelAuditOverview from './pages/LabelAuditOverview.jsx';
import AuditCaseExplorer from './pages/AuditCaseExplorer.jsx';
import ModelComparison from './pages/ModelComparison.jsx';

const linkClass = ({ isActive }) => `rounded-full px-4 py-2 text-sm font-semibold transition ${isActive ? 'bg-slate-900 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950'}`;

function App() {
  return <div className="dashboard-shell min-h-screen text-slate-900">
    <header className="app-header border-b border-slate-200 bg-white/90 backdrop-blur"><div className="mx-auto flex max-w-7xl items-center justify-between gap-5 px-5 py-4 sm:px-8">
      <NavLink to="/" className="flex items-center gap-3" aria-label="AgentTriage home"><span className="brand-mark grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-500 text-lg font-extrabold text-white">A</span><span><b className="block text-base tracking-tight">AgentTriage</b><small className="block text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">Defect prediction</small></span></NavLink>
      <nav className="flex items-center rounded-full border border-slate-200 bg-slate-50 p-1" aria-label="Main navigation"><NavLink end to="/" className={linkClass}>Dataset Insights</NavLink><NavLink to="/predict" className={linkClass}>Predict an Issue</NavLink><NavLink to="/audit" className={linkClass}>Label Audit</NavLink><NavLink to="/audit/cases" className={linkClass}>Audited Cases</NavLink><NavLink to="/compare" className={linkClass}>Model Comparison</NavLink></nav>
    </div></header>
    <main className="mx-auto max-w-7xl px-5 py-9 sm:px-8 sm:py-12"><Routes><Route path="/" element={<DatasetInsights />} /><Route path="/predict" element={<PredictIssue />} /><Route path="/audit" element={<LabelAuditOverview />} /><Route path="/audit/cases" element={<AuditCaseExplorer />} /><Route path="/compare" element={<ModelComparison />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></main>
  </div>;
}
export default App;
