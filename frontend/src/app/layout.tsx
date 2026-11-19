import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'WELLWATCH PREVENT — Life-Course Preventive Care Intelligence',
  description: 'Precision prevention, screening gap detection, and preventive-care navigation for Abu Dhabi, UAE.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-screen bg-[#F8FAF9] text-slate-900 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
