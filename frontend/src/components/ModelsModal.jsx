import React from 'react';
import { X, CheckCircle, AlertTriangle, Cpu, Layers, HardDrive, ShieldCheck } from 'lucide-react';

export default function ModelsModal({ isOpen, onClose, modelsData }) {
  if (!isOpen) return null;

  const models = modelsData?.models || [];
  const device = modelsData?.device || {};

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: 'rgba(0, 0, 0, 0.75)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '1rem'
    }}>
      <div className="glass-panel" style={{
        width: '100%',
        maxWidth: '900px',
        maxHeight: '90vh',
        overflowY: 'auto',
        backgroundColor: '#0f172a',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8)',
        padding: '1.75rem',
        borderRadius: '16px'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          borderBottom: '1px solid var(--border-subtle)',
          paddingBottom: '1rem',
          marginBottom: '1.25rem'
        }}>
          <div>
            <h2 style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.4rem',
              fontWeight: 700,
              color: '#ffffff',
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem'
            }}>
              <Layers size={22} color="#38bdf8" />
              Model Architecture & Registry
            </h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Centralized registry (`models/registry.yaml`) with dynamic CPU memory throttling (Max {modelsData?.max_loaded_models || 2} loaded simultaneously).
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '0.25rem'
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Hardware Status Banner */}
        <div style={{
          background: 'rgba(56, 189, 248, 0.08)',
          border: '1px solid rgba(56, 189, 248, 0.2)',
          borderRadius: '8px',
          padding: '0.85rem 1.1rem',
          marginBottom: '1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          fontSize: '0.825rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Cpu size={16} color="#38bdf8" />
            <span>Target Hardware: <strong>{device.resolved?.toUpperCase() || 'CPU'}</strong></span>
            <span style={{ color: 'var(--text-faint)' }}>({device.reason || 'Auto-configured for system CPU'})</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <HardDrive size={16} color="#10b981" />
            <span>Torch Threads: <strong>{device.torch_threads || 16}</strong></span>
          </div>
        </div>

        {/* Model Cards List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {models.map(m => (
            <div key={m.id} style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: m.loaded ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid var(--border-subtle)',
              borderRadius: '10px',
              padding: '1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.6rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                    {m.name}
                  </h3>
                  <span className="badge badge-slate" style={{ fontSize: '0.7rem' }}>
                    {m.id}
                  </span>
                  <span className={`badge ${m.kind === 'ml' ? 'badge-cyan' : 'badge-amber'}`}>
                    {m.kind.toUpperCase()}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {m.loaded && (
                    <span className="badge badge-emerald">
                      <CheckCircle size={11} /> Loaded in RAM
                    </span>
                  )}
                  {m.available ? (
                    <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>
                      Ready
                    </span>
                  ) : (
                    <span className="badge badge-amber" style={{ fontSize: '0.7rem' }}>
                      <AlertTriangle size={11} /> Standby / Download on demand
                    </span>
                  )}
                </div>
              </div>

              {/* Specs & Metadata */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                gap: '0.5rem',
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                marginTop: '0.2rem'
              }}>
                <div>Backend: <span style={{ color: '#e2e8f0', fontFamily: 'var(--font-mono)' }}>{m.backend}</span></div>
                <div>Memory: <span style={{ color: '#e2e8f0' }}>{m.size_mb ? `${m.size_mb} MB` : 'N/A'}</span></div>
                <div>License: <span style={{ color: '#e2e8f0' }}>{m.license || 'Permissive'}</span></div>
                <div>Score Calibration: <span style={{ color: m.provides_scores ? '#10b981' : '#f59e0b' }}>{m.provides_scores ? 'Calibrated Model Scores' : 'Heuristic / Unavailable'}</span></div>
              </div>

              {/* Tasks Covered */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-faint)' }}>Tasks:</span>
                {m.tasks.map(t => (
                  <span key={t} style={{
                    fontSize: '0.68rem',
                    background: 'rgba(56, 189, 248, 0.08)',
                    color: '#38bdf8',
                    padding: '0.1rem 0.45rem',
                    borderRadius: '4px',
                    fontFamily: 'var(--font-mono)'
                  }}>
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
