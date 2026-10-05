import React from 'react';
export default function LabelBadge({ label }) { const bug = label === 'Bug'; return <span className={`inline-flex whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-bold ${bug ? 'bg-rose-50 text-rose-700' : 'bg-emerald-50 text-emerald-700'}`}>{label}</span>; }
