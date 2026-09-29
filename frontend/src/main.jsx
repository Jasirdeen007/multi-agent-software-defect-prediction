import React, { Component, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App.jsx';
import './styles.css';

class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) { return { error }; }

  componentDidCatch(error) { console.error('AgentTriage render error:', error); }

  render() {
    if (!this.state.error) return this.props.children;
    return <main className="grid min-h-screen place-items-center bg-slate-50 p-6 text-center"><section className="max-w-md rounded-2xl border border-slate-200 bg-white p-8 shadow-sm"><p className="text-xs font-bold uppercase tracking-[0.16em] text-indigo-600">AgentTriage</p><h1 className="mt-3 text-2xl font-bold tracking-tight">The dashboard could not load.</h1><p className="mt-3 text-sm leading-6 text-slate-600">Refresh this page once. If the issue persists, open the browser console and share the error shown there.</p></section></main>;
  }
}

createRoot(document.getElementById('root')).render(<StrictMode><ErrorBoundary><BrowserRouter><App /></BrowserRouter></ErrorBoundary></StrictMode>);
