import React from 'react';

export default function SampleDataBanner() {
  return (
    <div className="mt-5 rounded-xl border border-indigo-100 bg-indigo-50 px-4 py-3 text-sm text-indigo-900">
      <b>Pipeline preview.</b> This screen shows the final product flow. Pending KG, agent-audit, and treatment outputs will populate here after those stages run.
    </div>
  );
}
