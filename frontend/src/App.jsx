import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ModelsModal from './components/ModelsModal';
import UploadZone from './components/UploadZone';
import QueryBar from './components/QueryBar';
import AnswerCard from './components/AnswerCard';
import VisualEvidence from './components/VisualEvidence';
import EvidenceTable from './components/EvidenceTable';
import ExecutionTrace from './components/ExecutionTrace';
import { generatePresetImageFile } from './demoPresets';
import { AlertCircle, Compass } from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

export default function App() {
  const [health, setHealth] = useState(null);
  const [modelsData, setModelsData] = useState(null);
  const [isModelsModalOpen, setIsModelsModalOpen] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(true);

  const [selectedFiles, setSelectedFiles] = useState([]);
  const [activePresetId, setActivePresetId] = useState(null);
  const [query, setQuery] = useState('');
  const [taskOverride, setTaskOverride] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [response, setResponse] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [activeEvidenceId, setActiveEvidenceId] = useState(null);

  const timerRef = useRef(null);
  const resultsRef = useRef(null);

  // Load health & models info on startup
  useEffect(() => {
    fetchHealth();
    fetchModels();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
        setIsDemoMode(data.demo_mode);
      }
    } catch (e) {
      console.warn('Backend /health unreachable:', e);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/models`);
      if (res.ok) {
        const data = await res.json();
        setModelsData(data);
      }
    } catch (e) {
      console.warn('Backend /models unreachable:', e);
    }
  };

  // Handle Preset Selection
  const handleApplyPreset = async (preset) => {
    setActivePresetId(preset.id);
    setQuery(preset.query);
    setTaskOverride(preset.task);
    setErrorMsg(null);

    // Generate real synthetic sample image files matching the preset roles
    const generated = [];
    for (let i = 0; i < preset.sampleImages.length; i++) {
      const sample = preset.sampleImages[i];
      const imgObj = await generatePresetImageFile(sample.name, sample.role, i + 1);
      generated.push(imgObj);
    }
    setSelectedFiles(generated);
  };

  // Run Analysis Handler
  const handleAnalyze = async (overrideTask = null) => {
    if (!query.trim() || selectedFiles.length === 0) {
      setErrorMsg('Please enter a natural-language query and ensure at least one satellite image is loaded.');
      return;
    }

    setIsLoading(true);
    setElapsedTime(0);
    setErrorMsg(null);
    setResponse(null);

    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      setElapsedTime((Date.now() - startTime) / 1000);
    }, 100);

    try {
      const formData = new FormData();
      formData.append('query', query.trim());

      const chosenTask = overrideTask || taskOverride;
      if (chosenTask) {
        formData.append('task', chosenTask);
      }

      // Append files and their corresponding roles
      selectedFiles.forEach((item) => {
        formData.append('files', item.file);
        formData.append('roles', item.role || 'optical');
      });

      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      });

      let data;
      const text = await res.text();
      try {
        data = JSON.parse(text);
      } catch (jsonErr) {
        throw new Error(`Server returned HTTP ${res.status}: ${text.slice(0, 120) || 'Backend service unreachable'}`);
      }

      if (!res.ok && !data.execution_trace) {
        throw new Error(data.error?.message || `HTTP ${res.status}: Failed to analyze`);
      }

      setResponse(data);

      // Auto-scroll down to results on completion
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 200);
    } catch (err) {
      console.error('Analysis error:', err);
      setErrorMsg(err.message || 'Network error occurred while connecting to GeoSense AI backend.');
    } finally {
      setIsLoading(false);
      clearInterval(timerRef.current);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Header */}
      <Header
        health={health}
        onOpenModels={() => setIsModelsModalOpen(true)}
        isDemoMode={isDemoMode}
        setIsDemoMode={setIsDemoMode}
      />

      {/* Models Modal */}
      <ModelsModal
        isOpen={isModelsModalOpen}
        onClose={() => setIsModelsModalOpen(false)}
        modelsData={modelsData}
      />

      {/* Main Body */}
      <main style={{
        flex: 1,
        maxWidth: '1440px',
        width: '100%',
        margin: '0 auto',
        padding: '0 1.25rem 2.5rem 1.25rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem'
      }}>
        {/* Error Alert */}
        {errorMsg && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.12)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '10px',
            padding: '0.85rem 1.25rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            color: '#fca5a5',
            fontSize: '0.85rem'
          }}>
            <AlertCircle size={18} />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Input & Control Panel */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Upload Zone & Presets */}
          <UploadZone
            selectedFiles={selectedFiles}
            setSelectedFiles={setSelectedFiles}
            onApplyPreset={handleApplyPreset}
            activePresetId={activePresetId}
          />

          {/* Natural Language Query Bar */}
          <QueryBar
            query={query}
            setQuery={setQuery}
            onAnalyze={() => handleAnalyze()}
            isLoading={isLoading}
            elapsedTime={elapsedTime}
            taskOverride={taskOverride}
            setTaskOverride={setTaskOverride}
            canSubmit={selectedFiles.length > 0 && query.trim().length > 0}
          />
        </section>

        {/* Results Section */}
        {response && (
          <section ref={resultsRef} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '0.5rem' }}>
            {/* Synthesized Answer with Transparent Confidence */}
            <AnswerCard
              response={response}
              onSelectClarification={(opt) => handleAnalyze(opt)}
            />

            {/* Split Grid: Visual Evidence Canvas (Left) + Evidence Table (Right) */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(0, 1.35fr) minmax(0, 1fr)',
              gap: '1.25rem',
              alignItems: 'start'
            }}>
              <VisualEvidence
                response={response}
                activeEvidenceId={activeEvidenceId}
                setActiveEvidenceId={setActiveEvidenceId}
              />

              <EvidenceTable
                evidence={response.evidence}
                result={response.result}
                activeEvidenceId={activeEvidenceId}
                setActiveEvidenceId={setActiveEvidenceId}
              />
            </div>

            {/* Transparent Execution Trace */}
            <ExecutionTrace
              trace={response.execution_trace}
              processingTime={response.processing_time}
              device={response.device}
            />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer style={{
        textAlign: 'center',
        padding: '1.5rem',
        fontSize: '0.75rem',
        color: 'var(--text-faint)',
        borderTop: '1px solid var(--border-subtle)'
      }}>
        GeoSense AI • Smart India Hackathon (SIH 2026) Problem Statement SIH26167: SatQuery AI • VoiceTheftAuto
      </footer>
    </div>
  );
}
