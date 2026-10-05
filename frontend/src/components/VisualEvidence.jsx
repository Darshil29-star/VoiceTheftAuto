import React, { useState, useRef } from 'react';
import { Layers, Eye, EyeOff, Sliders, Maximize2, Crosshair, MapPin } from 'lucide-react';

export default function VisualEvidence({
  response,
  activeEvidenceId,
  setActiveEvidenceId
}) {
  const [showBBoxes, setShowBBoxes] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
  const [activeVisIndex, setActiveVisIndex] = useState(0);
  const [sliderPos, setSliderPos] = useState(50); // for before-after comparison
  const [cursorCoords, setCursorCoords] = useState(null);

  const containerRef = useRef(null);

  if (!response) return null;

  const visualizations = response.visualizations || [];
  const evidence = response.evidence || [];
  const images = response.images || [];

  // Extract spatial regions / bounding boxes from evidence
  const spatialRegions = evidence.filter(e => e.type === 'spatial_region' || e.bbox);

  // Check if we have bi-temporal before / after visualizations
  const beforeVis = visualizations.find(v => v.title.toLowerCase().includes('before') || v.id.includes('before'));
  const afterVis = visualizations.find(v => v.title.toLowerCase().includes('after') || v.id.includes('after'));
  const changeVis = visualizations.find(v => v.kind === 'change_map' || v.title.toLowerCase().includes('change'));

  const isComparisonMode = beforeVis && (afterVis || changeVis);

  // Active visualization to render
  const currentVis = visualizations[activeVisIndex] || visualizations[0];

  // Handle cursor movement to calculate pixel coordinates & geo coordinates
  const handleMouseMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const widthPct = Math.max(0, Math.min(1, x / rect.width));
    const heightPct = Math.max(0, Math.min(1, y / rect.height));

    const img = images[0] || { width: 1024, height: 1024 };
    const pxX = Math.round(widthPct * (currentVis?.width || img.width));
    const pxY = Math.round(heightPct * (currentVis?.height || img.height));

    let geoCoord = null;
    if (img.transform && img.transform.length >= 6) {
      // Affine transform: X_geo = c + a*pxX + b*pxY, Y_geo = f + d*pxX + e*pxY
      const [a, b, c, d, e, f] = img.transform;
      const geoX = (c + a * pxX + b * pxY).toFixed(2);
      const geoY = (f + d * pxX + e * pxY).toFixed(2);
      geoCoord = `${geoX}, ${geoY} (${img.crs || 'CRS'})`;
    }

    setCursorCoords({ pxX, pxY, geoCoord });
  };

  return (
    <div className="glass-panel" style={{
      display: 'flex',
      flexDirection: 'column',
      borderRadius: '12px',
      overflow: 'hidden'
    }}>
      {/* Top Controls Toolbar */}
      <div style={{
        padding: '0.75rem 1rem',
        background: 'rgba(15, 23, 42, 0.8)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.6rem'
      }}>
        {/* Layer Selector tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
          <Layers size={15} color="#38bdf8" />
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>Layers:</span>
          {visualizations.map((vis, idx) => (
            <button
              key={vis.id || idx}
              onClick={() => setActiveVisIndex(idx)}
              style={{
                background: activeVisIndex === idx ? 'rgba(56, 189, 248, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                border: activeVisIndex === idx ? '1px solid #38bdf8' : '1px solid var(--border-subtle)',
                color: activeVisIndex === idx ? '#38bdf8' : 'var(--text-muted)',
                borderRadius: '6px',
                padding: '0.2rem 0.6rem',
                fontSize: '0.72rem',
                cursor: 'pointer',
                fontWeight: activeVisIndex === idx ? 600 : 400
              }}
            >
              {vis.title || vis.kind}
            </button>
          ))}
        </div>

        {/* Toggles */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {spatialRegions.length > 0 && (
            <button
              onClick={() => setShowBBoxes(!showBBoxes)}
              className="btn-secondary"
              style={{ fontSize: '0.72rem', padding: '0.2rem 0.55rem' }}
              title="Toggle Grounding Bounding Box Overlays"
            >
              {showBBoxes ? <Eye size={12} color="#10b981" /> : <EyeOff size={12} />}
              <span>Boxes ({spatialRegions.length})</span>
            </button>
          )}

          <div style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-faint)' }}>
            {currentVis ? `${currentVis.width}×${currentVis.height}px` : ''}
          </div>
        </div>
      </div>

      {/* Canvas Area */}
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setCursorCoords(null)}
        style={{
          position: 'relative',
          width: '100%',
          minHeight: '440px',
          maxHeight: '600px',
          backgroundColor: '#050811',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          userSelect: 'none'
        }}
      >
        {/* Normal Visualization Render */}
        {currentVis ? (
          <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', justifyContent: 'center' }}>
            <img
              src={
                currentVis.url?.startsWith('http') || !import.meta.env.VITE_API_URL
                  ? currentVis.url
                  : `${import.meta.env.VITE_API_URL.replace(/\/$/, '')}${currentVis.url}`
              }
              alt={currentVis.title}
              style={{
                maxWidth: '100%',
                maxHeight: '560px',
                objectFit: 'contain',
                display: 'block'
              }}
            />

            {/* SVG Overlay for Bounding Boxes and Polygons */}
            {showBBoxes && spatialRegions.length > 0 && (
              <svg
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none'
                }}
                viewBox={`0 0 ${currentVis.width || 1024} ${currentVis.height || 1024}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {spatialRegions.map((region) => {
                  if (!region.bbox || region.bbox.length < 4) return null;
                  const [x0, y0, x1, y1] = region.bbox;
                  const w = x1 - x0;
                  const h = y1 - y0;
                  const isHighlighted = activeEvidenceId === region.id;
                  const conf = region.confidence?.value ? ` ${(region.confidence.value * 100).toFixed(0)}%` : '';

                  return (
                    <g key={region.id} style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                       onClick={() => setActiveEvidenceId(isHighlighted ? null : region.id)}>
                      {/* Bounding Rectangle */}
                      <rect
                        x={x0}
                        y={y0}
                        width={w}
                        height={h}
                        fill={isHighlighted ? 'rgba(56, 189, 248, 0.25)' : 'rgba(56, 189, 248, 0.08)'}
                        stroke={isHighlighted ? '#38bdf8' : '#0284c7'}
                        strokeWidth={isHighlighted ? 3 : 1.75}
                        strokeDasharray={isHighlighted ? 'none' : '4,2'}
                      />

                      {/* Tag pill above box */}
                      <rect
                        x={x0}
                        y={Math.max(0, y0 - 18)}
                        width={Math.max(60, (region.label.length + conf.length) * 7.5 + 10)}
                        height={18}
                        fill="#0f172a"
                        opacity={0.88}
                        rx={3}
                      />
                      <text
                        x={x0 + 4}
                        y={Math.max(12, y0 - 5)}
                        fill="#38bdf8"
                        fontSize={11}
                        fontFamily="var(--font-mono)"
                        fontWeight="600"
                      >
                        {region.label}{conf}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            No visual evidence layer generated for this query.
          </div>
        )}

        {/* HUD / Coordinate readout */}
        {cursorCoords && (
          <div style={{
            position: 'absolute',
            bottom: '10px',
            left: '12px',
            background: 'rgba(15, 23, 42, 0.85)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: '6px',
            padding: '0.25rem 0.6rem',
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            color: '#38bdf8',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            pointerEvents: 'none'
          }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <Crosshair size={12} /> X: {cursorCoords.pxX} Y: {cursorCoords.pxY} px
            </span>
            {cursorCoords.geoCoord && (
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#10b981' }}>
                <MapPin size={12} /> {cursorCoords.geoCoord}
              </span>
            )}
          </div>
        )}

        {/* Legend Overlay if provided */}
        {currentVis?.legend && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '12px',
            background: 'rgba(15, 23, 42, 0.85)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '0.4rem 0.65rem',
            fontSize: '0.7rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem'
          }}>
            <div style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.65rem', textTransform: 'uppercase' }}>
              Legend
            </div>
            {Object.entries(currentVis.legend).map(([label, color]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: color }} />
                <span style={{ color: '#e2e8f0' }}>{label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Before / After Comparison Slider if in bi-temporal comparison mode */}
      {isComparisonMode && (
        <div style={{
          padding: '0.65rem 1rem',
          background: 'rgba(15, 23, 42, 0.9)',
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem'
        }}>
          <Sliders size={14} color="#38bdf8" />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Comparison Blend:</span>
          <input
            type="range"
            min="0"
            max="100"
            value={sliderPos}
            onChange={(e) => setSliderPos(Number(e.target.value))}
            style={{ flex: 1, accentColor: '#38bdf8', cursor: 'pointer' }}
          />
          <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
            {sliderPos}% T2 / Change
          </span>
        </div>
      )}
    </div>
  );
}
