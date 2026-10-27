import React from 'react';

const meanings = {
  Accuracy: 'Overall share of correct predictions.',
  'Balanced Accuracy': 'Accuracy that gives both classes equal weight.',
  'Macro F1': 'Average F1 across bug and non-bug classes.',
  'ROC-AUC': 'How well the model separates the two labels.',
  MCC: 'A balanced quality score for two-class predictions.',
  'Brier Score': 'How close confidence probabilities are to outcomes.',
};

export default function MetricsTable({ metrics, treatmentStatus = 'unavailable' }) {
  const treatmentUnavailable = treatmentStatus === 'blocked' ? 'Blocked' : 'Unavailable';
  return (
    <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white shadow-sm">
      <table className="w-full min-w-[620px] text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th className="p-4">Metric</th>
            <th className="p-4">Original-label baseline</th>
            <th className="p-4">Audited-label treatment</th>
            <th className="p-4">Difference</th>
          </tr>
        </thead>
        <tbody>{metrics.map((metric) => {
          const hasBaseline = typeof metric.baseline === 'number';
          const hasTreatment = typeof metric.treatment === 'number';
          const hasBoth = hasBaseline && hasTreatment;
          const change = hasBoth ? metric.treatment - metric.baseline : null;
          const better = hasBoth && (metric.lowerIsBetter ? change < 0 : change > 0);
          return (
            <tr key={metric.key} className="border-t border-slate-100">
              <td className="p-4 font-semibold text-slate-800"><span title={meanings[metric.label]} className="cursor-help border-b border-dotted border-slate-400">{metric.label}</span></td>
              <td className="p-4 text-slate-600">{hasBaseline ? metric.baseline.toFixed(3) : 'Unavailable'}</td>
              <td className="p-4 text-slate-500">{hasTreatment ? metric.treatment.toFixed(3) : treatmentUnavailable}</td>
              <td className={`p-4 font-semibold ${better ? 'text-emerald-700' : 'text-slate-500'}`}>{hasBoth ? `${change >= 0 ? '+' : '-'} ${Math.abs(change).toFixed(3)}` : treatmentUnavailable}</td>
            </tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}
