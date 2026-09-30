import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Upload, ShieldCheck, Activity, Clock3, BarChart3, AlertCircle } from 'lucide-react';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function Metric({ icon: Icon, label, value }) {
  return <div className="metric"><Icon size={20}/><div><span>{label}</span><strong>{value}</strong></div></div>;
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => setHealth(null));
    fetch(`${API}/api/experiments`).then(r => r.json()).then(x => setExperiments(x.experiments || [])).catch(() => {});
  }, []);

  function selectFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null); setError('');
    setPreview(URL.createObjectURL(f));
  }

  async function predict() {
    if (!file) return;
    setBusy(true); setError(''); setResult(null);
    const form = new FormData(); form.append('file', file);
    try {
      const r = await fetch(`${API}/api/predict`, { method: 'POST', body: form });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return <div className="app">
    <header className="header">
      <div><div className="brand"><ShieldCheck/> DeepGuard</div><p>Digital Media Authenticity Detection System</p></div>
      <div className={`status ${health?.model_available ? 'online' : 'offline'}`}><span/> {health?.model_available ? 'Model ready' : 'Model checkpoint required'}</div>
    </header>

    <main>
      <section className="hero"><div><span className="eyebrow">IMAGE-BASED FORENSICS</span><h1>Detect facial deepfakes with a reproducible research pipeline.</h1><p>Upload a facial image to classify it as real or deepfake using the verified production checkpoint.</p></div></section>

      <section className="grid">
        <div className="panel upload-panel">
          <div className="panel-title"><Upload/> <h2>Image analysis</h2></div>
          <label className="dropzone">
            <input type="file" accept="image/*" onChange={selectFile}/>
            {preview ? <img src={preview} alt="Selected"/> : <><Upload size={34}/><strong>Choose a facial image</strong><span>PNG, JPG or JPEG</span></>}
          </label>
          <button disabled={!file || busy || !health?.model_available} onClick={predict}>{busy ? 'Analyzing…' : 'Analyze image'}</button>
          {error && <div className="error"><AlertCircle size={18}/>{error}</div>}
        </div>

        <div className="panel result-panel">
          <div className="panel-title"><Activity/> <h2>Prediction</h2></div>
          {!result ? <div className="empty">Prediction results will appear here.</div> : <div className="result">
            <div className={`prediction ${result.label}`}><span>{result.label === 'real' ? 'REAL' : 'DEEPFAKE'}</span><strong>{(result.confidence * 100).toFixed(1)}%</strong></div>
            <div className="prob"><div><span>Deepfake</span><b>{(result.probabilities.deepfake * 100).toFixed(1)}%</b></div><div className="bar"><i style={{width: `${result.probabilities.deepfake * 100}%`}}/></div><div><span>Real</span><b>{(result.probabilities.real * 100).toFixed(1)}%</b></div><div className="bar"><i style={{width: `${result.probabilities.real * 100}%`}}/></div></div>
            <div className="details"><Metric icon={Clock3} label="Inference" value={`${result.inference_time_seconds}s`}/><Metric icon={Activity} label="Device" value={result.device}/></div>
          </div>}
        </div>
      </section>

      <section className="panel research"><div className="panel-title"><BarChart3/><h2>Research experiments</h2></div>
        {experiments.length === 0 ? <div className="empty">No standardized experiment records found yet. Run the training pipeline to populate this table.</div> : <div className="table-wrap"><table><thead><tr><th>Run</th><th>Model</th><th>Validation</th><th>Test</th><th>F1</th><th>Time</th></tr></thead><tbody>{experiments.map((x,i)=><tr key={i}><td>{x.run_id || x.tag || `Run ${i+1}`}</td><td>{x.model || '—'}</td><td>{x.best_validation_accuracy != null ? `${(x.best_validation_accuracy*100).toFixed(2)}%` : '—'}</td><td>{x.test_accuracy != null ? `${(x.test_accuracy*100).toFixed(2)}%` : '—'}</td><td>{x.f1_macro != null ? `${(x.f1_macro*100).toFixed(2)}%` : '—'}</td><td>{x.training_time_seconds != null ? `${Number(x.training_time_seconds).toFixed(1)}s` : '—'}</td></tr>)}</tbody></table></div>}
      </section>
    </main>
    <footer>DeepGuard · Research prototype · Image-only facial deepfake detection</footer>
  </div>
}

createRoot(document.getElementById('root')).render(<App/>);
