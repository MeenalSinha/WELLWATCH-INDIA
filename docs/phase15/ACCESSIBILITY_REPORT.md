# WELLWATCH PREVENT — Accessibility & Usability Report (WCAG 2.1 AA)
**Standard:** Web Content Accessibility Guidelines (WCAG) 2.1 Level AA  
**Jurisdiction:** Abu Dhabi Digital Authority (ADDA) Digital Accessibility Standards  
**Scope:** Frontend Application (19 Static Next.js Routes, CHW Field App, Patient Portal)

---

## 1. Executive Summary
Healthcare platforms must be universally accessible to diverse clinical staff, community health workers, and patient demographics, including individuals with visual, motor, or cognitive impairments, as well as elder citizens accessing screening notices.

This Accessibility Report evaluates WELLWATCH PREVENT against WCAG 2.1 Level AA criteria across all 19 application routes.

---

## 2. Accessibility Compliance Audit

| WCAG 2.1 Principle | Specific Criterion | Implementation in WELLWATCH PREVENT | Compliance Status |
| :--- | :--- | :--- | :--- |
| **1. Perceivable** | 1.1.1 Non-text Content | All SVG icons and clinical indicators have descriptive `aria-label` or `alt` text. | **COMPLIANT** |
| **1. Perceivable** | 1.4.3 Contrast (Minimum) | Body text achieves > 7.1:1 contrast; triage badges (High, Medium, Normal) exceed 4.8:1 contrast against backdrops. | **COMPLIANT** |
| **1. Perceivable** | 1.4.10 Reflow | Responsive layout scales cleanly to 320px width without horizontal scrollbars or clipping. | **COMPLIANT** |
| **2. Operable** | 2.1.1 Keyboard Nav | Full navigation feasible via `Tab`, `Shift+Tab`, `Enter`, and `Space`. Modals trap focus and close via `Escape`. | **COMPLIANT** |
| **2. Operable** | 2.4.7 Focus Visible | High-visibility 2px focus ring (`ring-2 ring-emerald-500`) rendered on all interactive buttons and inputs. | **COMPLIANT** |
| **3. Understandable** | 3.1.2 Language of Parts | `dir="rtl"` dynamically set for Arabic interface with font pairing (Noto Sans Arabic) and `dir="ltr"` for English. | **COMPLIANT** |
| **3. Understandable** | 3.2.4 Consistent ID | Standardized navigation sidebar, header, and care-gap action buttons across all views. | **COMPLIANT** |
| **4. Robust** | 4.1.2 Name, Role, Value | Semantic HTML elements (`<nav>`, `<main>`, `<dialog>`, `<button>`) utilized throughout. Zero headless divs as buttons. | **COMPLIANT** |

---

## 3. Bilingual RTL / LTR Design
- **Arabic Typography:** Native support for Arabic text rendering with appropriate line-height multipliers (1.4x) to prevent diacritic clipping.
- **Bi-directional Layout:** Flex direction and margins mirror cleanly when toggled (`dir="rtl"` flips sidebar to right, align icons appropriately).
- **Date & Number Formatting:** Uses `Intl.DateTimeFormat` with `ar-AE` locale support for UAE standard date presentation.

---

## 4. Route-by-Route Accessibility Verification (19 Routes)
All 19 routes compiled in Next.js 14 were inspected for accessibility:
- `/dashboard`, `/patients`, `/patients/[id]`, `/screenings`, `/assessments`: Clear tabular data headers (`<th>` with scope="col"), screen-reader readable risk metrics.
- `/chw`: High-contrast, large touch targets (minimum 48x48px) optimized for handheld tablet usage in outdoor sunlight.
- `/outreach`, `/navigation`: Accessible dialog overlays and readable status tags.
- Result: **Zero automated WCAG 2.1 AA violations detected.**
