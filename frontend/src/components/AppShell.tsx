'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredUser } from '@/lib/auth';
import { TokenResponse } from '@/types';
import { Sidebar } from './Sidebar';
import { TopHeader } from './TopHeader';
import { LoadingSpinner } from './LoadingSpinner';

interface AppShellProps {
  children: React.ReactNode;
  activePage?: string;
  pageTitle?: string;
  breadcrumbs?: { label: string; href?: string }[];
}

export const AppShell: React.FC<AppShellProps> = ({
  children,
  pageTitle,
  breadcrumbs,
}) => {
  const router = useRouter();
  const [user, setUser] = useState<TokenResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  useEffect(() => {
    const activeUser = getStoredUser();
    if (!activeUser) {
      router.push('/login');
      return;
    }
    setUser(activeUser);
    setIsLoading(false);
  }, [router]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
        <LoadingSpinner size="lg" />
        <span className="text-xs font-semibold text-slate-500 mt-3">Verifying clinical session...</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex bg-[#F8FAF9] text-slate-900">
      {/* Collapsible Left Navigation Sidebar */}
      <Sidebar
        user={user}
        isOpenMobile={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <TopHeader
          user={user}
          onOpenMobileSidebar={() => setIsMobileSidebarOpen(true)}
          pageTitle={pageTitle}
          breadcrumbs={breadcrumbs}
        />

        <main className="flex-1 w-full max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          {children}
        </main>
      </div>
    </div>
  );
};

