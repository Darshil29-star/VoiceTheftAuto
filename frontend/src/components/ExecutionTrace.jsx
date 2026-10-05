import React, { useState } from 'react';
import { Terminal, CheckCircle2, AlertTriangle, AlertCircle, Clock, ChevronDown, ChevronUp, Cpu, Workflow } from 'lucide-react';

export default function ExecutionTrace({ trace = [], processingTime, device }) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState({});

  if (!trace || trace.length === 0) return null;

  const toggleStep = (idx) => {
    setExpandedSteps(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'ok':
        return { cls: 'badge-emerald', icon: CheckCircle2 };
      case 'warning':
      case 'fallback':
        return { cls: 'badge-amber', icon: AlertTriangle };
      case 'error':
        return { cls: 'badge-rose', icon: AlertCircle };
      default:
        return { cls: 'badge-slate', icon: Clock };
    }
  };

  return (
    <div className="glass-panel" style={{
      borderRadius: '12px',
      overflow: 'hidden',
      border: '1px solid var(--border-subtle)'
    }}>
      {/* Collapsible Header */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{
          padding: '0.85rem 1.25rem',
          background: 'rgba(15, 23, 42, 0.7)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          userSelect: 'none'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Workflow size={17} color="#38bdf8" />
          <h3 style={{ fontSize: '0.875rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
            Transparent Execution Trace ({trace.length} Pipeline Stages)
          </h3>
          <span className="badge badge-cyan" style={{ fontSize: '0.68rem' }}>
            {processingTime ? `${(processingTime * 1000).toFixed(0)} ms` : '< 1s'} Total
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
            Hardware: {device?.toUpperCase() || 'CPU'}
          </span>
          {isOpen ? <ChevronUp size={16} color="var(--text-muted)" /> : <ChevronDown size={16} color="var(--text-muted)" />}
        </div>
      </div>

      {/* Expanded Stages Timeline */}
      {isOpen && (
        <div style={{
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
          background: 'rgba(10, 14, 23, 0.95)'
        }}>
          {trace.map((step, idx) => {
            const { cls, icon: StatusIcon } = getStatusBadge(step.status);
            const isExpanded = !!expandedSteps[idx];
            const hasDetails = step.details && Object.keys(step.details).length > 0;

            return (
              <div
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid rgba(255, 255, 255, 0.05)',
                  borderRadius: '8px',
                  padding: '0.75rem 1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.4rem'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className={`badge ${cls}`} style={{ fontSize: '0.68rem', padding: '0.1rem 0.45rem' }}>
                      <StatusIcon size={11} /> {step.stage.toUpperCase()}
                    </span>
                    <span style={{ fontSize: '0.825rem', fontWeight: 500, color: '#e2e8f0' }}>
                      {step.message}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    {step.duration_ms !== undefined && step.duration_ms !== null && (
                      <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-faint)' }}>
                        {step.duration_ms.toFixed(1)} ms
                      </span>
                    )}
                    {hasDetails && (
                      <button
                        onClick={() => toggleStep(idx)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#38bdf8',
                          fontSize: '0.72rem',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.2rem'
                        }}
                      >
                        {isExpanded ? 'Hide info' : 'View details'}
                      </button>
                    )}
                  </div>
                </div>

                {/* Structured JSON Step Details */}
                {isExpanded && hasDetails && (
                  <pre style={{
                    marginTop: '0.4rem',
                    background: '#070b14',
                    border: '1px solid rgba(56, 189, 248, 0.2)',
                    borderRadius: '6px',
                    padding: '0.6rem 0.8rem',
                    fontSize: '0.72rem',
                    fontFamily: 'var(--font-mono)',
                    color: '#93c5fd',
                    overflowX: 'auto',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all'
                  }}>
                    {JSON.stringify(step.details, null, 2)}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
