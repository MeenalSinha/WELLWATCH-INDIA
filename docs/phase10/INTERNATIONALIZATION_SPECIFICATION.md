# WELLWATCH PREVENT — Internationalization (i18n) & RTL Specification

## 1. Bilingual Architecture
The platform is engineered with native bilingual capabilities adhering to UAE Federal standards:
- **Primary Language:** Modern Standard Arabic (`ar`).
- **Secondary Language:** English (`en`).
- **Text Directionality:**
  - Arabic: Right-to-Left (`dir="rtl"`).
  - English: Left-to-Right (`dir="ltr"`).

---

## 2. Dynamic Directionality & UI Rendering
In the frontend application (`frontend/src/app/outreach/page.tsx`), the language selector triggers dynamic direction switching:
```tsx
<div dir={activeLanguage === 'ar' ? 'rtl' : 'ltr'} className="p-6 bg-slate-900 border rounded-xl">
  <div className="flex items-center justify-between">
    <h3 className="font-semibold">{activeLanguage === 'ar' ? 'معاينة الرسالة الموجهة' : 'Outreach Message Preview'}</h3>
    <span className="badge">{activeLanguage.toUpperCase()} ({activeLanguage === 'ar' ? 'RTL' : 'LTR'})</span>
  </div>
  <p className="mt-4 text-slate-200">{renderedMessage}</p>
</div>
```

---

## 3. Cultural Etiquette & Tone
- **Arabic Tone:** Respectful, formal, and community-centric. Uses polite honorifics ("عزيزي / عزيزتي ولي الأمر", "نحن في خدمتكم").
- **Clinical Clarity:** Avoiding obtuse jargon in citizen-facing outreach, while maintaining clinical precision.
- **Wadeema's Law Respect:** Tone specifically respects parental and guardian roles in child health monitoring.
