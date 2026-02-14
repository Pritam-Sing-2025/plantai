import { useState, useRef } from 'react'
import './App.css'

// Icon components (simple SVG icons to replace lucide-react)
const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
);

const AlertIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
);

const CheckIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
);

const LayersIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
);

const XIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
);

const ActivityIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
);

const ZapIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
);

const LoaderIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="icon"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>
);

function App() {
  const [batchFiles, setBatchFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('Ensemble (All Models)');
  const [isBatchMode, setIsBatchMode] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      if (isBatchMode) {
        setBatchFiles(prev => [...prev, ...files]);
      } else {
        setBatchFiles(files);
        setResults([]);
      }
      setError(null);
    }
  };



  const handleAnalyze = async () => {
    if (batchFiles.length === 0) {
      setError('Please select/upload plant images.');
      return;
    }

    setLoading(true);
    setProgress(0);
    setError(null);
    const newResults = [];

    try {
      for (let i = 0; i < batchFiles.length; i++) {
        const file = batchFiles[i];
        const imageUrl = URL.createObjectURL(file);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_name', selectedModel);

        const response = await fetch('http://localhost:9101/predict', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Connection failed to port 9101');

        const result = await response.json();
        newResults.push({ fileName: file.name, imageUrl, ...result });
        setProgress(Math.round(((i + 1) / batchFiles.length) * 100));
      }
      setResults(newResults);
    } catch {
      setError('Server Error: Ensure the Python backend is running on http://localhost:9101');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-container">
      <div className="app-card">
        {/* Sidebar / Controls Section */}
        <div className="sidebar">
          {/* Header */}
          <div className="sidebar-header">
            <div className="logo-container">
              <div className="logo-emoji">🌿</div>
              <div>
                <h1 className="logo-title">PlantAI</h1>
                <p className="logo-tagline">Disease Detection & Analysis</p>
              </div>
            </div>
          </div>

          <div className="sidebar-content custom-scrollbar">
            {/* Enhanced Upload Container */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`upload-container ${batchFiles.length > 0 ? 'has-files' : ''}`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                multiple={isBatchMode}
                accept="image/*"
              />
              {loading ? (
                <div className="upload-loading">
                  <div className="loading-icon animate-spin"><LoaderIcon /></div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                  </div>
                  <span className="progress-text">{progress}% ANALYZED</span>
                </div>
              ) : (
                <>
                  <div className="upload-icon">
                    <UploadIcon />
                  </div>
                  <h3 className="upload-title">
                    {batchFiles.length > 0 ? `${batchFiles.length} Specimen(s)` : 'Upload Leaf'}
                  </h3>
                  <p className="upload-subtitle">Click to browse</p>
                </>
              )}
            </div>

            {/* Model Config */}
            <div className="form-group">
              <label className="form-label">
                <ActivityIcon /> Neural Architecture:
              </label>
              <select
                className="select-input"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option>Ensemble (All Models)</option>
                <option>EfficientNetV2 (High Res)</option>
                <option>ResNet50V2 (Deep Layers)</option>
                <option>MobileNetV3 (Speed Focus)</option>
              </select>
            </div>

            {/* Mode Switcher */}
            <div className="button-group">
              <button
                onClick={() => { setIsBatchMode(!isBatchMode); setBatchFiles([]); setResults([]); }}
                className={`mode-button ${isBatchMode ? 'active' : ''}`}
              >
                <LayersIcon /> {isBatchMode ? 'Batch On' : 'Single Mode'}
              </button>
              <button
                onClick={() => { setBatchFiles([]); setResults([]); }}
                className="reset-button"
                title="Reset"
              >
                <XIcon />
              </button>
            </div>

            {/* Primary Action */}
            <button
              onClick={handleAnalyze}
              disabled={loading || batchFiles.length === 0}
              className={`primary-button ${loading ? 'loading' : ''}`}
            >
              {loading ? 'PROCESSING...' : 'RUN DIAGNOSTICS'}
            </button>

            {/* Error Handler */}
            {error && (
              <div className="error-message animate-pulse">
                <div className="error-icon"><AlertIcon /></div>
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="sidebar-footer">
            <p className="footer-text">Kaggle Agricultural Lab • 2026</p>
          </div>
        </div>

        {/* Main Content / Results Section */}
        <div className="results-container">
          {/* Decorative Background Elements */}
          <div className="bg-decoration-1"></div>
          <div className="bg-decoration-2"></div>

          <div className="results-content custom-scrollbar">
            {results.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🔬</div>
                <h3 className="empty-title">Ready for Analysis</h3>
                <p className="empty-subtitle">No data generated yet</p>
              </div>
            ) : (
              <div className={`results-grid ${isBatchMode ? 'batch-mode' : ''}`}>
                {results.map((res, idx) => (
                  <div
                    key={idx}
                    className={`result-card animate-fade-in ${res.disease.toLowerCase().includes('healthy') ? 'healthy' : 'diseased'
                      }`}
                  >
                    {/* Landscape Card Layout */}
                    <div className="card-header">
                      {res.imageUrl && (
                        <div className="card-image">
                          <img src={res.imageUrl} alt="Specimen" />
                          <div className="image-overlay">
                            <div className="image-filename">{res.fileName}</div>
                          </div>
                        </div>
                      )}

                      <div className="card-info">
                        {/* Header Row */}
                        <div className="card-info-header">
                          <h3 className="plant-name">{res.plant}</h3>
                          <div className="confidence-box">
                            <div className="confidence-value">{res.accuracy}%</div>
                            <div className="confidence-label">Confidence</div>
                          </div>
                        </div>

                        <div className={`disease-name ${res.disease.toLowerCase().includes('healthy') ? 'healthy' : 'diseased'}`}>
                          {res.disease}
                        </div>

                        {/* Model Consensus Visualization */}
                        {res.confidence_breakdown && (
                          <div className="model-consensus">
                            <p className="consensus-label">
                              <ZapIcon /> Model Consensus
                            </p>
                            {Object.entries(res.confidence_breakdown).map(([model, score], mIdx) => (
                              <div key={mIdx} className="consensus-item">
                                <span className="consensus-model-name">{model}</span>
                                <div className="consensus-bar-bg">
                                  <div
                                    className={`consensus-bar-fill ${score > 90 ? 'high-confidence' : 'low-confidence'}`}
                                    style={{ width: `${score}%` }}
                                  ></div>
                                </div>
                                <span className="consensus-value">{Math.round(score)}%</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="card-body">
                      <p className="description-text">"{res.description}"</p>

                      <div className="treatment-box">
                        <div className="treatment-icon-wrapper">
                          <div className="treatment-icon"><CheckIcon /></div>
                        </div>
                        <div className="treatment-content">
                          <span className="treatment-label">Recommended Protocol</span>
                          <p className="treatment-text">{res.treatment}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

export default App;
