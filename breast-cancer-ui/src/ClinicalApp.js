import React, { useState, useEffect } from 'react';

const CLINICAL_FEATURE_NAMES = [
  // Image features (30 features)
  'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
  'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
  'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
  'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
  'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
  'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension',
  // Clinical features (3 features)
  'age', 'family_history', 'menopausal_status'
];

const FEATURE_DESCRIPTIONS = {
  'age': 'Patient age in years (18-100)',
  'family_history': 'Family history of breast cancer (0=No, 1=Yes)',
  'menopausal_status': 'Menopausal status (0=Pre-menopausal, 1=Post-menopausal)',
  'mean_radius': 'Mean radius of tumor cells',
  'mean_texture': 'Mean texture of tumor cells',
  'mean_perimeter': 'Mean perimeter of tumor cells',
  'mean_area': 'Mean area of tumor cells',
  'mean_smoothness': 'Mean smoothness of tumor cells',
  'mean_compactness': 'Mean compactness of tumor cells',
  'mean_concavity': 'Mean concavity of tumor cells',
  'mean_concave_points': 'Mean concave points of tumor cells',
  'mean_symmetry': 'Mean symmetry of tumor cells',
  'mean_fractal_dimension': 'Mean fractal dimension of tumor cells'
};

function ClinicalApp() {
  const [formData, setFormData] = useState(Array(CLINICAL_FEATURE_NAMES.length).fill(''));
  const [csvFile, setCsvFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [dark, setDark] = useState(true);
  const [modelInfo, setModelInfo] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  // Load model info on component mount
  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch('http://localhost:5000/model-info');
      if (response.ok) {
        const info = await response.json();
        setModelInfo(info);
      }
    } catch (err) {
      console.error('Failed to fetch model info:', err);
    }
  };

  const handleInputChange = (idx, value) => {
    const newData = [...formData];
    newData[idx] = value;
    setFormData(newData);
  };

  const handleFileChange = (e) => {
    setCsvFile(e.target.files[0]);
    setResult(null);
    setError(null);
    setSuccess(false);
  };

  const handleReset = () => {
    setFormData(Array(CLINICAL_FEATURE_NAMES.length).fill(''));
    setCsvFile(null);
    setResult(null);
    setError(null);
    setSuccess(false);
    setLoading(false);
  };

  const validateFormData = () => {
    const errors = [];
    
    // Validate clinical features
    const age = parseFloat(formData[30]); // age is at index 30
    if (age < 18 || age > 100) {
      errors.push('Age must be between 18 and 100 years');
    }
    
    const familyHistory = parseInt(formData[31]); // family_history is at index 31
    if (familyHistory !== 0 && familyHistory !== 1) {
      errors.push('Family history must be 0 (No) or 1 (Yes)');
    }
    
    const menopausalStatus = parseInt(formData[32]); // menopausal_status is at index 32
    if (menopausalStatus !== 0 && menopausalStatus !== 1) {
      errors.push('Menopausal status must be 0 (Pre-menopausal) or 1 (Post-menopausal)');
    }
    
    // Validate image features (should be positive)
    for (let i = 0; i < 30; i++) {
      const value = parseFloat(formData[i]);
      if (value <= 0) {
        errors.push(`${CLINICAL_FEATURE_NAMES[i]} must be positive`);
      }
    }
    
    return errors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError(null);
    setSuccess(false);
    setLoading(true);
    
    try {
      let response;
      
      if (csvFile) {
        const data = new FormData();
        data.append('file', csvFile);
        response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          body: data
        });
      } else {
        // Validate form data
        const validationErrors = validateFormData();
        if (validationErrors.length > 0) {
          setError(`Validation errors: ${validationErrors.join(', ')}`);
          setLoading(false);
          return;
        }
        
        if (formData.some(v => v === '')) {
          setError('Please fill in all fields or upload a CSV file.');
          setLoading(false);
          return;
        }
        
        const values = formData.map(Number);
        const json = {};
        CLINICAL_FEATURE_NAMES.forEach((name, idx) => {
          json[name] = values[idx];
        });
        
        response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(json)
        });
      }

      const data = await response.json();

      if (!response.ok) {
        setError(data.error || 'Prediction failed.');
        setSuccess(false);
      } else if (data.results) {
        setResult(data.results);
        setSuccess(true);
        
        // Add to prediction history
        const historyEntry = {
          id: data.prediction_id,
          timestamp: data.timestamp,
          results: data.results,
          input_type: csvFile ? 'CSV Upload' : 'Manual Input'
        };
        setPredictionHistory(prev => [historyEntry, ...prev.slice(0, 9)]); // Keep last 10
      } else {
        setError('Invalid response from backend.');
        setSuccess(false);
      }

      setLoading(false);
    } catch (err) {
      console.error(err);
      setError('An unexpected error occurred.');
      setSuccess(false);
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'High': return 'text-red-600 dark:text-red-400';
      case 'Medium': return 'text-yellow-600 dark:text-yellow-400';
      case 'Low': return 'text-green-600 dark:text-green-400';
      default: return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.85) return 'text-green-600 dark:text-green-400';
    if (confidence >= 0.70) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className={dark ? 'dark min-h-screen bg-gray-900' : 'min-h-screen bg-gray-100'}>
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Clinical Breast Cancer Prediction</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">AI-powered clinical decision support system</p>
          </div>
          <div className="flex gap-4">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="px-4 py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 transition"
            >
              {showHistory ? 'Hide History' : 'Show History'}
            </button>
            <button
              className="rounded px-3 py-1 text-sm font-medium border border-gray-500 dark:border-gray-400 text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
              onClick={() => setDark(d => !d)}
            >
              {dark ? '🌙 Dark' : '☀️ Light'}
            </button>
          </div>
        </div>

        {/* Model Info */}
        {modelInfo && (
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mb-6">
            <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Model Information</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-blue-700 dark:text-blue-300">Type:</span> {modelInfo.model_type}
              </div>
              <div>
                <span className="text-blue-700 dark:text-blue-300">Features:</span> {modelInfo.feature_count}
              </div>
              <div>
                <span className="text-blue-700 dark:text-blue-300">Version:</span> {modelInfo.version}
              </div>
              <div>
                <span className="text-blue-700 dark:text-blue-300">Training Date:</span> {modelInfo.training_date}
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Form */}
          <div className="lg:col-span-2">
            <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
              <div className="mb-4 flex flex-col sm:flex-row sm:items-center gap-4">
                <label className="block text-gray-800 dark:text-gray-200 font-medium">Upload CSV file:</label>
                <input 
                  type="file" 
                  accept=".csv,.xlsx" 
                  onChange={handleFileChange} 
                  disabled={loading}
                  className="block w-full sm:w-auto text-gray-900 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 rounded p-2" 
                />
              </div>
              <div className="text-center text-gray-600 dark:text-gray-400 mb-4">or enter features manually:</div>
              
              {/* Clinical Features Section */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Clinical Information</h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {CLINICAL_FEATURE_NAMES.slice(30).map((name, idx) => (
                    <div key={name} className="flex flex-col">
                      <label className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                        {name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        <span className="text-xs text-gray-500 ml-1">*</span>
                      </label>
                      <input
                        type="number"
                        step="any"
                        value={formData[idx + 30]}
                        onChange={e => handleInputChange(idx + 30, e.target.value)}
                        disabled={!!csvFile || loading}
                        required={!csvFile}
                        className="rounded p-2 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder={name === 'age' ? '18-100' : name === 'family_history' ? '0 or 1' : '0 or 1'}
                      />
                      <p className="text-xs text-gray-500 mt-1">{FEATURE_DESCRIPTIONS[name]}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Image Features Section */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Image Analysis Features</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  {CLINICAL_FEATURE_NAMES.slice(0, 30).map((name, idx) => (
                    <div key={name} className="flex flex-col">
                      <label className="text-xs text-gray-700 dark:text-gray-300 mb-1">
                        {name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        <span className="text-xs text-gray-500 ml-1">*</span>
                      </label>
                      <input
                        type="number"
                        step="any"
                        value={formData[idx]}
                        onChange={e => handleInputChange(idx, e.target.value)}
                        disabled={!!csvFile || loading}
                        required={!csvFile}
                        className="rounded p-2 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Positive value"
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex flex-col sm:flex-row gap-4 justify-end">
                <button
                  type="submit"
                  disabled={loading}
                  className="px-6 py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:bg-gray-500 transition"
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin h-5 w-5 mr-2 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                      </svg>Analyzing...
                    </span>
                  ) : 'Analyze'}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={loading}
                  className="px-6 py-2 rounded bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold hover:bg-gray-400 dark:hover:bg-gray-600 disabled:bg-gray-500 transition"
                >
                  Reset
                </button>
              </div>
            </form>

            {/* Error Display */}
            {error && (
              <div className="mb-6">
                <div className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-lg px-4 py-3 shadow-md">
                  <span className="font-bold">Error:</span> {error}
                </div>
              </div>
            )}

            {/* Results Display */}
            {success && result && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">Clinical Analysis Results</h2>
                {result.map((r, i) => (
                  <div key={i} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 mb-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Primary Assessment</h3>
                        <div className={`text-2xl font-bold ${r.prediction === 'Malignant' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                          {r.prediction}
                        </div>
                        <div className={`text-lg ${getConfidenceColor(r.confidence)}`}>
                          Confidence: {(r.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          Uncertainty: {(r.uncertainty * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Clinical Recommendations</h3>
                        <div className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                          {r.recommendation}
                        </div>
                        <div className="space-y-1">
                          <div className={`text-sm ${r.risk_factors.high_risk ? 'text-red-600 dark:text-red-400 font-semibold' : 'text-gray-600 dark:text-gray-400'}`}>
                            {r.risk_factors.high_risk ? '⚠️ High Risk' : '✓ Low Risk'}
                          </div>
                          <div className="text-xs text-gray-500">
                            Benign Probability: {(r.probabilities.benign * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500">
                            Malignant Probability: {(r.probabilities.malignant * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <p className="text-sm text-yellow-800 dark:text-yellow-200">
                    <strong>Medical Disclaimer:</strong> This analysis is for clinical decision support only. 
                    Final diagnosis and treatment decisions should be made by qualified healthcare professionals 
                    based on comprehensive clinical evaluation.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar - Prediction History */}
          {showHistory && (
            <div className="lg:col-span-1">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Prediction History</h3>
                {predictionHistory.length === 0 ? (
                  <p className="text-gray-500 dark:text-gray-400 text-sm">No predictions yet</p>
                ) : (
                  <div className="space-y-3">
                    {predictionHistory.map((entry) => (
                      <div key={entry.id} className="border border-gray-200 dark:border-gray-700 rounded p-3">
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                          {new Date(entry.timestamp).toLocaleString()}
                        </div>
                        <div className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                          {entry.results[0]?.prediction}
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          Confidence: {(entry.results[0]?.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {entry.input_type}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ClinicalApp;




