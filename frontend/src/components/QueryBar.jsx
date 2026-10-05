import React from 'react';
import { Send, Loader2, Sparkles, Filter, CornerDownLeft } from 'lucide-react';
import { QUICK_PROMPTS } from '../demoPresets';

export const TASK_OPTIONS = [
  { value: '', label: 'Auto (Agent Intent Router)' },
  { value: 'captioning', label: 'Scene Captioning' },
  { value: 'grounding', label: 'Visual Grounding' },
  { value: 'vqa', label: 'Visual Q&A (Count/Inspect)' },
  { value: 'change_analysis', label: 'Bi-Temporal Change' },
  { value: 'optical_sar_fusion', label: 'Optical–SAR Fusion' },
];

export default function QueryBar({
  query,
  setQuery,
  onAnalyze,
  isLoading,
  elapsedTime,
  taskOverride,
  setTaskOverride,
  canSubmit
}) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canSubmit && !isLoading) {
        onAnalyze();
      }
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Top row: Task selector & Options */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sparkles size={15} color="#38bdf8" />
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f8fafc' }}>
            Natural-Language Query
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Filter size={13} color="var(--text-faint)" />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Routing:</span>
          <select
            value={taskOverride}
            onChange={(e) => setTaskOverride(e.target.value)}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '6px',
              color: '#e2e8f0',
              padding: '0.25rem 0.5rem',
              fontSize: '0.75rem',
              outline: 'none',
              cursor: 'pointer'
            }}
          >
            {TASK_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value} style={{ background: '#0f172a', color: '#fff' }}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main input & Send Button */}
      <div style={{ display: 'flex', gap: '0.75rem', position: 'relative' }}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question or issue a remote sensing instruction (e.g. 'Describe this scene', 'Count all fuel tanks', 'What changed between these dates?')..."
          rows={2}
          style={{
            flex: 1,
            background: 'var(--bg-input)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '8px',
            color: '#ffffff',
            padding: '0.75rem',
            fontSize: '0.875rem',
            fontFamily: 'inherit',
            resize: 'none',
            outline: 'none',
            lineHeight: 1.4,
            transition: 'border-color 0.2s ease'
          }}
          onFocus={(e) => e.target.style.borderColor = 'var(--accent-cyan)'}
          onBlur={(e) => e.target.style.borderColor = 'var(--border-subtle)'}
        />

        <button
          onClick={onAnalyze}
          disabled={!canSubmit || isLoading}
          className="btn-primary"
          style={{
            minWidth: '140px',
            flexDirection: 'column',
            gap: '0.2rem',
            padding: '0.5rem 1rem'
          }}
        >
          {isLoading ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Loader2 size={16} className="pulse-glow" style={{ animation: 'spin 1s linear infinite' }} />
                <span>Analyzing</span>
              </div>
              <span style={{ fontSize: '0.7rem', opacity: 0.8, fontFamily: 'var(--font-mono)' }}>
                {elapsedTime.toFixed(1)}s
              </span>
            </>
          ) : (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Send size={15} />
                <span>Run Analysis</span>
              </div>
              <span style={{ fontSize: '0.65rem', opacity: 0.7 }}>
                Enter ↵
              </span>
            </>
          )}
        </button>
      </div>

      {/* Quick Prompt Suggestions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-faint)' }}>Suggestions:</span>
        {QUICK_PROMPTS.map((prompt, i) => (
          <button
            key={i}
            onClick={() => setQuery(prompt)}
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '6px',
              padding: '0.15rem 0.5rem',
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#38bdf8';
              e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = 'var(--text-muted)';
              e.currentTarget.style.borderColor = 'var(--border-subtle)';
            }}
          >
            "{prompt}"
          </button>
        ))}
      </div>
    </div>
  );
}
