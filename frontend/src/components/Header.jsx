import React from 'react';
import { Satellite, Cpu, Database, Sparkles, ShieldAlert, Activity } from 'lucide-react';

export default function Header({ health, onOpenModels, isDemoMode, setIsDemoMode }) {
  return (
    <header className="glass-panel" style={{
      margin: '1.25rem 1.25rem 1rem 1.25rem',
      padding: '1rem 1.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1rem',
      position: 'relative',
      zIndex: 10
    }}>
      {/* Brand & SIH Badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
        <div style={{
          width: '44px',
          height: '44px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #0284c7, #3b82f6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 20px rgba(56, 189, 248, 0.4)'
        }}>
          <Satellite size={24} color="#ffffff" />
        </div>

        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <h1 style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.45rem',
              fontWeight: 700,
              letterSpacing: '-0.02em',
              background: 'linear-gradient(to right, #ffffff, #93c5fd)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              margin: 0
            }}>
              GeoSense AI
            </h1>
            <span className="badge badge-cyan">SIH26167: SatQuery AI</span>
            <span className="badge badge-slate" style={{ fontSize: '0.7rem' }}>Team: VoiceTheftAuto</span>
          </div>
          <p style={{
            margin: '0.2rem 0 0 0',
            fontSize: '0.8rem',
            color: 'var(--text-muted)'
          }}>
            Multimodal Remote-Sensing Vision-Language Intelligence System
          </p>
        </div>
      </div>

      {/* Telemetry & Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        {/* Device Status */}
        <div className="glass-panel" style={{
          padding: '0.35rem 0.75rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)'
        }}>
          <Cpu size={14} color="#38bdf8" />
          <span style={{ color: 'var(--text-muted)' }}>Device:</span>
          <span style={{
            color: health?.device?.toLowerCase().includes('cuda') ? '#10b981' : '#f59e0b',
            fontWeight: 600
          }}>
            {health?.device?.toUpperCase() || 'CPU (AUTO)'}
          </span>
        </div>

        {/* Models Registry Button */}
        <button
          onClick={onOpenModels}
          className="btn-secondary"
          style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}
          title="Inspect Registered Model Architecture"
        >
          <Database size={14} color="#8b5cf6" />
          <span>Models ({health?.models_available ?? 5}/{health?.models_enabled ?? 5})</span>
        </button>

        {/* Demo Mode Toggle */}
        <button
          onClick={() => setIsDemoMode(!isDemoMode)}
          className={`badge ${isDemoMode ? 'badge-emerald' : 'badge-slate'}`}
          style={{
            cursor: 'pointer',
            padding: '0.35rem 0.75rem',
            border: isDemoMode ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid var(--border-subtle)',
            background: isDemoMode ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)'
          }}
          title="Toggle Demo Preset Fallback Mode"
        >
          <Sparkles size={13} />
          <span>Demo Fallback: {isDemoMode ? 'ENABLED' : 'OFF'}</span>
        </button>

        {/* System Health Pulse */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }} title="Server API Status">
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: health?.status === 'ok' ? '#10b981' : '#f59e0b',
            display: 'inline-block',
            boxShadow: health?.status === 'ok' ? '0 0 10px #10b981' : '0 0 10px #f59e0b'
          }} />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>
            v{health?.version || '1.0.0'}
          </span>
        </div>
      </div>
    </header>
  );
}
