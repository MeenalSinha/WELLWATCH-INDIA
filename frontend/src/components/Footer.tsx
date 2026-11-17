import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-slate-200 mt-auto py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-xs text-slate-500">
          <div>
            <div className="font-semibold text-slate-900 mb-2">WELLWATCH PREVENT</div>
            <p className="leading-relaxed">
              AI-powered life-course preventive care intelligence platform engineered for
              the healthcare system of Abu Dhabi, UAE.
            </p>
          </div>
          <div>
            <div className="font-semibold text-slate-900 mb-2">Clinical Governance</div>
            <p className="leading-relaxed">
              Aligned with Department of Health Abu Dhabi (DoH) Child Health Guidelines and the
              IFHAS Periodic Adult Comprehensive Screening Program.
            </p>
          </div>
          <div>
            <div className="font-semibold text-slate-900 mb-2">Security & Sovereignty</div>
            <p className="leading-relaxed">
              Engineered under Abu Dhabi Healthcare Information and Cyber Security (ADHICS)
              standards and UAE Federal Law No. 2 of 2019.
            </p>
          </div>
        </div>
        <div className="mt-8 pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-400">
          <div>© 2026 WELLWATCH PREVENT. All rights reserved.</div>
          <div className="mt-2 sm:mt-0">
            Future Health Digital Innovation Challenge 2026 · Precision Prevention and Screening
          </div>
        </div>
      </div>
    </footer>
  );
};
