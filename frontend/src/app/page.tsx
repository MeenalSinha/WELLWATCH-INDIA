'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  ShieldCheck,
  Activity,
  HeartPulse,
  Brain,
  GitBranch,
  Navigation,
  BarChart3,
  CheckCircle2,
  Lock,
  ArrowRight,
  Database,
  Building2,
  Stethoscope,
  Sparkles,
} from 'lucide-react';
import { apiClient } from '@/lib/api';
import { HealthStatus } from '@/types';
import { StatusBadge } from '@/components/StatusBadge';

export default function HomePage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiClient
      .getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false));
  }, []);

  const capabilities = [
    {
      icon: Activity,
      title: 'Individual Health Intelligence',
      badge: 'Longitudinal Continuity',
      desc: 'Aggregates clinical observations, biometric vitals, family histories, and verified pediatric/adult profiles across the entire life-course.',
    },
    {
      icon: Brain,
      title: 'Explainable Preventive Risk Intelligence',
      badge: 'Epistemic Safety',
      desc: 'Transparent, evidence-backed assessment of cardiometabolic and developmental risks with explicit uncertainty tracking and epistemic abstention.',
    },
    {
      icon: GitBranch,
      title: 'Preventive Pathway Engine',
      badge: 'DoH & IFHAS Aligned',
      desc: 'Automates screening eligibility and care-gap detection under Abu Dhabi Child Health and Adult Comprehensive Screening Standards.',
    },
    {
      icon: Navigation,
      title: 'Care Navigation & Follow-Up',
      badge: 'Closed-Loop Care',
      desc: 'Orchestrates specialist referrals, automated screening recalls, and clinical care-gap closure verified exclusively against verifiable evidence.',
    },
    {
      icon: BarChart3,
      title: 'Population Health Intelligence',
      badge: 'Sovereign Analytics',
      desc: 'Provides population-level surveillance, geographic equity mapping, and small-cell privacy suppression strictly aligned with ADHICS.',
    },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAF9] text-slate-900">
      {/* UAE Jurisdiction Ribbon */}
      <div className="bg-forest-950 text-slate-300 text-[11px] px-6 py-1.5 flex items-center justify-between font-mono border-b border-forest-900">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Target Market: Abu Dhabi, UAE (DoH Guidelines / ADHICS Aligned)</span>
        </div>
        <div className="hidden sm:block text-slate-400">
          Future Health Digital Innovation Challenge 2026
        </div>
      </div>

      {/* Public Navigation Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200/80 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-forest-800 text-white flex items-center justify-center font-bold text-lg shadow-sm">
              W
            </div>
            <div className="leading-tight">
              <span className="font-extrabold text-sm tracking-tight text-slate-900">WELLWATCH</span>{' '}
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-forest-50 text-forest-700 border border-forest-200">
                PREVENT
              </span>
              <div className="text-[10px] text-slate-500 font-medium">
                Life-Course Preventive Care Intelligence
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/login"
              className="px-4 py-2 rounded-lg bg-forest-800 hover:bg-forest-900 text-white font-semibold text-xs shadow-xs transition flex items-center gap-1.5"
            >
              Sign In to Portal
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden py-16 sm:py-24 border-b border-slate-200/60 bg-gradient-to-b from-white via-white to-forest-50/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-forest-50 border border-forest-100 text-forest-800 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              <span>Precision Prevention & Population Screening Intelligence</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.1]">
              From Health Signals to{' '}
              <span className="text-forest-700 underline decoration-emerald-500 decoration-wavy decoration-2">
                Preventive Action
              </span>
              .
            </h1>

            <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-2xl">
              An intelligent preventive healthcare platform connecting longitudinal health insights,
              personalized screening pathways, explainable risk intelligence, and coordinated care navigation
              engineered for the healthcare system of Abu Dhabi, United Arab Emirates.
            </p>

            <div className="pt-2 flex flex-wrap gap-3.5 items-center">
              <Link
                href="/login"
                className="px-6 py-3 rounded-xl bg-forest-800 hover:bg-forest-900 text-white font-semibold text-sm shadow-sm transition flex items-center gap-2"
              >
                Access Healthcare Portal
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                href="/dashboard"
                className="px-6 py-3 rounded-xl bg-white hover:bg-slate-50 text-slate-800 font-semibold text-sm border border-slate-200 shadow-2xs transition"
              >
                Explore Live Platform
              </Link>
            </div>

            {/* Quick Proof Pills */}
            <div className="pt-4 flex flex-wrap items-center gap-4 text-xs text-slate-500">
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Child Health & Developmental Milestones</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>DoH IFHAS Adult Screening Standards</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>ADHICS Sovereign Data Protection</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Live Operational Telemetry Bar */}
      <section className="py-8 bg-white border-b border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Live Backend Telemetry
              </span>
              <h2 className="text-base font-bold text-slate-900">Platform Operational Baseline</h2>
            </div>
            {health && <StatusBadge status={health.status} />}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl border border-slate-200/90 bg-[#F8FAF9]">
              <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                <span>FastAPI Service</span>
                <Activity className="w-4 h-4 text-forest-700" />
              </div>
              <div className="text-xl font-bold text-slate-900">
                {loading ? 'Verifying...' : health?.status || 'Offline'}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {health ? `Version ${health.version}` : 'Connecting...'}
              </div>
            </div>

            <div className="p-4 rounded-xl border border-slate-200/90 bg-[#F8FAF9]">
              <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                <span>Database Engine</span>
                <Database className="w-4 h-4 text-forest-700" />
              </div>
              <div className="text-xl font-bold text-slate-900">
                {loading ? 'Verifying...' : health?.database === 'connected' ? 'Connected' : 'Unavailable'}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">PostgreSQL 16 High-Availability</div>
            </div>

            <div className="p-4 rounded-xl border border-slate-200/90 bg-[#F8FAF9]">
              <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                <span>Target Jurisdiction</span>
                <Building2 className="w-4 h-4 text-forest-700" />
              </div>
              <div className="text-xl font-bold text-slate-900">Abu Dhabi, UAE</div>
              <div className="text-[11px] text-slate-500 mt-1">DoH Guidelines & UAE Federal Law No. 2</div>
            </div>

            <div className="p-4 rounded-xl border border-slate-200/90 bg-[#F8FAF9]">
              <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
                <span>Clinical Pathways</span>
                <HeartPulse className="w-4 h-4 text-forest-700" />
              </div>
              <div className="text-xl font-bold text-slate-900">Active Care Rules</div>
              <div className="text-[11px] text-slate-500 mt-1">IFHAS Adult + Well-Child Visits</div>
            </div>
          </div>
        </div>
      </section>

      {/* Core Innovation Pillars */}
      <section className="py-16 sm:py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <span className="text-[11px] font-semibold text-forest-700 uppercase tracking-wider bg-forest-50 px-2.5 py-1 rounded-full border border-forest-100">
            Precision Prevention Architecture
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mt-3">
            Five Pillars of Preventive Healthcare Intelligence
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-2">
            Engineered to transition healthcare from reactive acute encounters toward continuous, proactive, and equitable population wellness.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {capabilities.map((cap, idx) => {
            const Icon = cap.icon;
            return (
              <div
                key={idx}
                className="bg-white p-6 rounded-2xl border border-slate-200/90 shadow-xs hover:shadow-sm hover:border-slate-300 transition duration-150 flex flex-col justify-between"
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="w-10 h-10 rounded-xl bg-forest-50 border border-forest-100 text-forest-800 flex items-center justify-center font-bold">
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">
                      {cap.badge}
                    </span>
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-slate-900 tracking-tight">{cap.title}</h3>
                    <p className="text-xs text-slate-500 mt-2 leading-relaxed">{cap.desc}</p>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Interactive Portal Card */}
          <div className="bg-gradient-to-br from-forest-900 to-forest-950 p-6 rounded-2xl text-white shadow-xs flex flex-col justify-between">
            <div className="space-y-3">
              <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                Authorized Access
              </span>
              <h3 className="text-base font-bold text-white tracking-tight">
                Healthcare Professionals & Citizens Portal
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Log in as Dr. Mariam Al Mansoori (Physician), Community Health Worker, Care Coordinator, or Citizen to explore live pathways.
              </p>
            </div>
            <div className="pt-6">
              <Link
                href="/login"
                className="w-full py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-forest-950 font-bold text-xs shadow-xs transition flex items-center justify-center gap-1.5"
              >
                Launch Portal
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Refined Footer */}
      <footer className="mt-auto bg-white border-t border-slate-200 py-12 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-3 md:col-span-2">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-forest-800 text-white flex items-center justify-center font-bold text-sm">
                W
              </div>
              <span className="font-extrabold text-sm text-slate-900 tracking-tight">WELLWATCH PREVENT</span>
            </div>
            <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
              AI-powered life-course preventive care intelligence platform engineered for the healthcare system of Abu Dhabi, UAE.
            </p>
            <div className="text-[11px] text-slate-400">
              Future Health Digital Innovation Challenge 2026 — Precision Prevention & Screening
            </div>
          </div>

          <div className="space-y-2">
            <div className="font-semibold text-slate-800 uppercase tracking-wider text-[11px]">
              Clinical Governance
            </div>
            <div className="space-y-1.5 text-xs text-slate-500">
              <div>DoH Abu Dhabi Child Health Guidelines</div>
              <div>IFHAS Periodic Adult Comprehensive Screening</div>
              <div>UAE National Immunization Program</div>
              <div>Wadeema's Law (Federal Law No. 3 of 2016)</div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="font-semibold text-slate-800 uppercase tracking-wider text-[11px]">
              Security & Sovereignty
            </div>
            <div className="space-y-1.5 text-xs text-slate-500">
              <div>ADHICS v2.0 Standard Compliance</div>
              <div>UAE Health Data Law (Federal Law No. 2 of 2019)</div>
              <div>Role-Based Access Control (8 Personas)</div>
              <div>Strict Epistemic Evidence Verifiability</div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
