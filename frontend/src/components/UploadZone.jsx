import React, { useRef } from 'react';
import { UploadCloud, Image, Trash2, Tag, Compass, Sparkles, Layers } from 'lucide-react';
import { DEMO_PRESETS } from '../demoPresets';

export const IMAGE_ROLES = [
  { value: 'optical', label: 'Optical (RGB/MSI)', color: '#38bdf8' },
  { value: 'sar', label: 'SAR (Radar)', color: '#8b5cf6' },
  { value: 'before', label: 'T1 Before', color: '#10b981' },
  { value: 'after', label: 'T2 After', color: '#f59e0b' }
];

export default function UploadZone({
  selectedFiles,
  setSelectedFiles,
  onApplyPreset,
  activePresetId
}) {
  const fileInputRef = useRef(null);

  const handleFilesAdded = (filesList) => {
    const newFiles = Array.from(filesList).map((file, idx) => {
      // Default role assignment heuristic based on existing files or filename
      let defaultRole = 'optical';
      const fname = file.name.toLowerCase();
      if (fname.includes('sar') || fname.includes('grd') || fname.includes('radar')) {
        defaultRole = 'sar';
      } else if (fname.includes('before') || fname.includes('t1') || fname.includes('pre')) {
        defaultRole = 'before';
      } else if (fname.includes('after') || fname.includes('t2') || fname.includes('post')) {
        defaultRole = 'after';
      } else if (selectedFiles.length === 1 && selectedFiles[0].role === 'before') {
        defaultRole = 'after';
      } else if (selectedFiles.length === 1 && selectedFiles[0].role === 'optical') {
        defaultRole = 'sar';
      }

      return {
        file,
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        preview: URL.createObjectURL(file),
        role: defaultRole,
        isPreset: false
      };
    });

    setSelectedFiles(prev => [...prev, ...newFiles].slice(0, 2)); // Limit to 2 for multi-temporal/fusion
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesAdded(e.dataTransfer.files);
    }
  };

  const handleRoleChange = (index, newRole) => {
    setSelectedFiles(prev => prev.map((item, i) => i === index ? { ...item, role: newRole } : item));
  };

  const handleRemove = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Demo Scenario Preset Pills */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
          <Sparkles size={14} color="#38bdf8" />
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Quick Demo Scenarios (SIH 2026 Test Cases)
          </span>
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.5rem'
        }}>
          {DEMO_PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => onApplyPreset(preset)}
              style={{
                background: activePresetId === preset.id ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                border: activePresetId === preset.id ? '1px solid #38bdf8' : '1px solid var(--border-subtle)',
                borderRadius: '8px',
                padding: '0.6rem 0.75rem',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f8fafc' }}>
                  {preset.title}
                </span>
                <span className="badge badge-slate" style={{ fontSize: '0.65rem', padding: '0.1rem 0.35rem' }}>
                  {preset.category}
                </span>
              </div>
              <p style={{
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
                margin: 0,
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden'
              }}>
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Dropzone & Preview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedFiles.length > 0 ? '1fr 1fr' : '1fr', gap: '0.85rem' }}>
        {/* Dropzone (always accessible unless 2 files uploaded) */}
        {selectedFiles.length < 2 && (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="glass-panel"
            style={{
              padding: '1.5rem',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              border: '2px dashed rgba(56, 189, 248, 0.3)',
              borderRadius: '12px',
              cursor: 'pointer',
              minHeight: '160px',
              background: 'rgba(15, 22, 36, 0.5)'
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => handleFilesAdded(e.target.files)}
              multiple
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              style={{ display: 'none' }}
            />
            <UploadCloud size={32} color="#38bdf8" style={{ marginBottom: '0.6rem' }} />
            <div style={{ fontSize: '0.875rem', fontWeight: 600, color: '#e2e8f0' }}>
              Upload Satellite Imagery
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Drag & drop GeoTIFF / TIFF, PNG or JPEG (up to 2 images for bi-temporal / fusion)
            </div>
          </div>
        )}

        {/* Selected Images Cards */}
        {selectedFiles.map((item, idx) => (
          <div
            key={idx}
            className="glass-panel"
            style={{
              padding: '0.85rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.6rem',
              borderRadius: '12px',
              position: 'relative'
            }}
          >
            {/* Top row: Name & Remove */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', overflow: 'hidden' }}>
                <Image size={15} color="#38bdf8" />
                <span style={{
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                  overflow: 'hidden',
                  maxWidth: '180px'
                }}>
                  {item.name}
                </span>
                <span className="badge badge-slate" style={{ fontSize: '0.65rem' }}>{item.size || 'Sample'}</span>
              </div>
              <button
                onClick={() => handleRemove(idx)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-faint)',
                  cursor: 'pointer',
                  padding: '0.2rem'
                }}
                title="Remove image"
              >
                <Trash2 size={15} />
              </button>
            </div>

            {/* Thumbnail Preview */}
            <div style={{
              width: '100%',
              height: '110px',
              borderRadius: '8px',
              overflow: 'hidden',
              backgroundColor: '#000',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative'
            }}>
              {item.name?.toLowerCase().endsWith('.tif') || item.name?.toLowerCase().endsWith('.tiff') ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '0.4rem',
                  color: '#38bdf8'
                }}>
                  <Layers size={28} />
                  <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>GeoTIFF Raster</span>
                </div>
              ) : (
                <img
                  src={item.preview}
                  alt={item.name}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              )}
              {item.meta && (
                <div style={{
                  position: 'absolute',
                  bottom: '4px',
                  right: '6px',
                  background: 'rgba(0,0,0,0.7)',
                  padding: '0.15rem 0.4rem',
                  borderRadius: '4px',
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#38bdf8'
                }}>
                  {item.meta.crs || 'GeoTIFF'} • {item.meta.gsd || 'Multi-band'}
                </div>
              )}
            </div>

            {/* Role Assignment Pill Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Role:</span>
              {IMAGE_ROLES.map(r => (
                <button
                  key={r.value}
                  onClick={() => handleRoleChange(idx, r.value)}
                  style={{
                    background: item.role === r.value ? `${r.color}25` : 'rgba(255, 255, 255, 0.05)',
                    border: item.role === r.value ? `1px solid ${r.color}` : '1px solid var(--border-subtle)',
                    color: item.role === r.value ? r.color : 'var(--text-muted)',
                    borderRadius: '6px',
                    padding: '0.2rem 0.5rem',
                    fontSize: '0.7rem',
                    cursor: 'pointer',
                    fontWeight: item.role === r.value ? 600 : 400,
                    transition: 'all 0.15s ease'
                  }}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
