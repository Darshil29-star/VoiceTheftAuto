import React, { useState } from 'react';
import {
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Cpu,
  Info,
  ChevronDown,
  ChevronUp,
  Layers,
  AlertTriangle,
  Sparkles
} from 'lucide-react';

export default function AnswerCard({ response, onSelectClarification }) {
  const [showConfidenceDetails, setShowConfidenceDetails] = useState(false);

  if (!response) return null;

  const {
    status,
    answer,
    confidence,
    confidence_note,
    models_used,
    warnings,
    clarification,
    processing_time,
    device,
    demo_fallback
  } = response;

  // Determine badge color and label for confidence
  const getConfidenceBadge = () => {
    if (!confidence || confidence.level === 'unavailable') {
      return {
        cls: 'badge-slate',
        label: confidence_note || 'Confidence: Unavailable from model',
        icon: HelpCircle
      };
    }
    const pct = confidence.value !== null && confidence.value !== undefined ? `(${Math.round(confidence.value * 100)}%)` : '';
    if (confidence.level === 'high') {
      return { cls: 'badge-emerald', label: `High Confidence ${pct}`, icon: CheckCircle2 };
    }
    if (confidence.level === 'medium') {
      return { cls: 'badge-amber', label: `Medium Confidence ${pct}`, icon: Info };
    }
    return { cls: 'badge-rose', label: `Low Confidence ${pct}`, icon: AlertCircle };
  };

  const badgeInfo = getConfidenceBadge();
  const BadgeIcon = badgeInfo.icon;

  return (
    <div className="glass-panel animate-fade-in" style={{
      padding: '1.25rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      borderLeft: '4px solid var(--accent-cyan)'
    }}>
      {/* Top Banner: Status, Confidence & Model Tags */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.6rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{
            fontSize: '0.75rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--text-muted)'
          }}>
            Analysis Output
          </span>

          {/* Transparent Confidence Badge */}
          <div
            onClick={() => setShowConfidenceDetails(!showConfidenceDetails)}
            className={`badge ${badgeInfo.cls}`}
            style={{ cursor: 'pointer', userSelect: 'none' }}
            title="Click to view confidence calculation breakdown"
          >
            <BadgeIcon size={12} />
            <span>{badgeInfo.label}</span>
            {showConfidenceDetails ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </div>

          {/* Demo Fallback Badge */}
          {demo_fallback && (
            <span className="badge badge-amber" title="Inference completed using deterministic offline fallback pipeline">
              <Sparkles size={11} /> Demo Fallback
            </span>
          )}
        </div>

        {/* Latency & Device stats */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-faint)' }}>
          <span>{processing_time ? `${(processing_time * 1000).toFixed(0)} ms` : '< 1s'}</span>
          <span>•</span>
          <span>{device?.toUpperCase() || 'CPU'}</span>
        </div>
      </div>

      {/* Clarification Needed Banner */}
      {status === 'clarification_needed' && clarification && (
        <div style={{
          background: 'rgba(245, 158, 11, 0.12)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '8px',
          padding: '0.85rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.6rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#f59e0b', fontWeight: 600, fontSize: '0.85rem' }}>
            <HelpCircle size={16} />
            <span>Clarification Needed: {clarification.question}</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {clarification.options?.map((opt) => (
              <button
                key={opt}
                onClick={() => onSelectClarification && onSelectClarification(opt)}
                className="btn-secondary"
                style={{ fontSize: '0.75rem', padding: '0.3rem 0.7rem' }}
              >
                Proceed with {opt.replace('_', ' ').toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Primary Synthesized Answer */}
      <div style={{
        fontSize: '0.975rem',
        lineHeight: 1.6,
        color: '#f8fafc',
        whiteSpace: 'pre-wrap',
        fontWeight: 400
      }}>
        {answer || (status === 'error' ? 'An error occurred during analysis.' : 'No textual answer returned.')}
      </div>

      {/* Expandable Transparent Confidence Breakdown */}
      {showConfidenceDetails && confidence && (
        <div className="glass-panel" style={{
          padding: '0.85rem 1rem',
          backgroundColor: 'rgba(15, 23, 42, 0.8)',
          fontSize: '0.8rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.4rem',
          border: '1px solid rgba(56, 189, 248, 0.2)'
        }}>
          <div style={{ fontWeight: 600, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Info size={14} /> Confidence Attribution & Evaluation Formula
          </div>
          <div><strong>Origin:</strong> {confidence.source || 'Specialist Engine'} ({confidence.kind})</div>
          <div><strong>Calibration:</strong> {confidence.calibrated ? 'Empirically Calibrated' : 'Uncalibrated / Raw Detector Logits'}</div>
          <div><strong>Rationale:</strong> {confidence.explanation}</div>
          {confidence.factors && confidence.factors.length > 0 && (
            <div style={{ marginTop: '0.2rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Underlying Factors:</span>
              <ul style={{ margin: '0.2rem 0 0 1.25rem', color: 'var(--text-faint)' }}>
                {confidence.factors.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Warnings & Caveats Callout */}
      {warnings && warnings.length > 0 && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid rgba(239, 68, 68, 0.2)',
          borderRadius: '8px',
          padding: '0.65rem 0.85rem',
          fontSize: '0.78rem',
          color: '#fca5a5',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.25rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600 }}>
            <AlertTriangle size={14} />
            <span>Geospatial / Pipeline Caveats:</span>
          </div>
          <ul style={{ margin: '0 0 0 1.25rem' }}>
            {warnings.map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Footer: Models Used Tags */}
      {models_used && models_used.length > 0 && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          flexWrap: 'wrap',
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '0.65rem',
          marginTop: '0.2rem'
        }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-faint)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <Layers size={12} /> Models Utilized:
          </span>
          {models_used.map((m, idx) => (
            <span key={idx} className="badge badge-slate" style={{ fontSize: '0.68rem' }} title={`Backend: ${m.backend}, Kind: ${m.kind}`}>
              {m.name} ({m.version})
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
