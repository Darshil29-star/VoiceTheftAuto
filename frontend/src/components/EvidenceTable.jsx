import React from 'react';
import { Table, MapPin, CheckCircle, BarChart3, HelpCircle, Layers } from 'lucide-react';

export default function EvidenceTable({
  evidence = [],
  result = {},
  activeEvidenceId,
  setActiveEvidenceId
}) {
  const spatialItems = evidence.filter(e => e.type === 'spatial_region' || e.bbox);
  const counts = result.counts || {};
  const changeSummary = result.change_summary || {};
  const fusionMetrics = result.consensus || {};

  return (
    <div className="glass-panel" style={{
      padding: '1rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem'
    }}>
      {/* Table Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Table size={16} color="#38bdf8" />
          <h3 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
            Structured Evidence & Geospatial Measurements
          </h3>
        </div>
        <span className="badge badge-slate" style={{ fontSize: '0.7rem' }}>
          {spatialItems.length} Spatial Feature{spatialItems.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Aggregate Counts Banner if present */}
      {Object.keys(counts).length > 0 && (
        <div style={{
          display: 'flex',
          gap: '0.6rem',
          flexWrap: 'wrap',
          background: 'rgba(56, 189, 248, 0.05)',
          padding: '0.65rem 0.85rem',
          borderRadius: '8px',
          border: '1px solid rgba(56, 189, 248, 0.15)'
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <BarChart3 size={13} color="#38bdf8" /> Object Counts:
          </div>
          {Object.entries(counts).map(([cls, cnt]) => (
            <div key={cls} className="badge badge-cyan" style={{ fontSize: '0.72rem' }}>
              <strong>{cls}:</strong> {cnt}
            </div>
          ))}
        </div>
      )}

      {/* Bi-Temporal Change Metrics if present */}
      {result.change_pct !== undefined && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '0.6rem',
          background: 'rgba(244, 63, 94, 0.06)',
          padding: '0.75rem',
          borderRadius: '8px',
          border: '1px solid rgba(244, 63, 94, 0.2)'
        }}>
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Changed Area</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f43f5e', fontFamily: 'var(--font-mono)' }}>
              {result.change_pct.toFixed(2)}%
            </div>
          </div>
          {result.change_area_m2 && (
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Ground Footprint</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f8fafc', fontFamily: 'var(--font-mono)' }}>
                {(result.change_area_m2 / 10000).toFixed(2)} ha ({result.change_area_m2.toLocaleString()} m²)
              </div>
            </div>
          )}
          {result.alignment_score !== undefined && (
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ECC Registration</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#10b981', fontFamily: 'var(--font-mono)' }}>
                {(result.alignment_score * 100).toFixed(1)}% match
              </div>
            </div>
          )}
        </div>
      )}

      {/* Optical-SAR Consensus Metrics if present */}
      {result.agreement_rate !== undefined && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '0.6rem',
          background: 'rgba(139, 92, 246, 0.08)',
          padding: '0.75rem',
          borderRadius: '8px',
          border: '1px solid rgba(139, 92, 246, 0.2)'
        }}>
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Cross-Modal Agreement</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#a78bfa', fontFamily: 'var(--font-mono)' }}>
              {(result.agreement_rate * 100).toFixed(1)}%
            </div>
          </div>
          {result.cloud_penetration_pct !== undefined && (
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Cloud Penetration Recovery</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                {result.cloud_penetration_pct.toFixed(1)}%
              </div>
            </div>
          )}
        </div>
      )}

      {/* Features Table */}
      {spatialItems.length > 0 ? (
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.775rem',
            textAlign: 'left'
          }}>
            <thead>
              <tr style={{
                borderBottom: '1px solid var(--border-subtle)',
                color: 'var(--text-muted)',
                fontSize: '0.7rem',
                textTransform: 'uppercase',
                letterSpacing: '0.04em'
              }}>
                <th style={{ padding: '0.45rem 0.6rem' }}>Feature / Label</th>
                <th style={{ padding: '0.45rem 0.6rem' }}>Source Engine</th>
                <th style={{ padding: '0.45rem 0.6rem' }}>Confidence</th>
                <th style={{ padding: '0.45rem 0.6rem' }}>Pixel Bounding Box</th>
                <th style={{ padding: '0.45rem 0.6rem' }}>Ground Area</th>
              </tr>
            </thead>
            <tbody>
              {spatialItems.map((item) => {
                const isSelected = activeEvidenceId === item.id;
                const bboxStr = item.bbox ? `[${item.bbox.map(v => Math.round(v)).join(', ')}]` : 'N/A';
                const areaStr = item.geo?.area_m2
                  ? `${Math.round(item.geo.area_m2).toLocaleString()} m²`
                  : item.area_px
                    ? `${Math.round(item.area_px)} px²`
                    : 'N/A';

                return (
                  <tr
                    key={item.id}
                    onClick={() => setActiveEvidenceId(isSelected ? null : item.id)}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                      backgroundColor: isSelected ? 'rgba(56, 189, 248, 0.12)' : 'transparent',
                      cursor: 'pointer',
                      transition: 'background-color 0.15s ease'
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <td style={{ padding: '0.5rem 0.6rem', fontWeight: 600, color: '#f8fafc' }}>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                        <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#38bdf8' }} />
                        {item.label}
                      </span>
                    </td>
                    <td style={{ padding: '0.5rem 0.6rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {item.source || 'Specialist'}
                    </td>
                    <td style={{ padding: '0.5rem 0.6rem' }}>
                      {item.confidence?.value !== null && item.confidence?.value !== undefined ? (
                        <span className={`badge ${item.confidence.value >= 0.75 ? 'badge-emerald' : item.confidence.value >= 0.5 ? 'badge-amber' : 'badge-slate'}`} style={{ fontSize: '0.68rem', padding: '0.1rem 0.4rem' }}>
                          {(item.confidence.value * 100).toFixed(0)}%
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-faint)', fontSize: '0.7rem' }}>N/A</span>
                      )}
                    </td>
                    <td style={{ padding: '0.5rem 0.6rem', fontFamily: 'var(--font-mono)', color: 'var(--text-faint)', fontSize: '0.72rem' }}>
                      {bboxStr}
                    </td>
                    <td style={{ padding: '0.5rem 0.6rem', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                      {areaStr}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-faint)', textAlign: 'center', padding: '1rem' }}>
          No discrete spatial bounding features in current query result.
        </div>
      )}
    </div>
  );
}
