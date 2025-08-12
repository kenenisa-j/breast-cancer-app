import React, { useState } from 'react';
import Papa from 'papaparse';
// Feature definitions with units and descriptions
const FEATURE_GROUPS = {
  'Mean Values': [
    { name: 'radius_mean', label: 'Mean Radius', unit: 'mm', placeholder: '17.99' },
    { name: 'texture_mean', label: 'Mean Texture', unit: '', placeholder: '10.38' },
    { name: 'perimeter_mean', label: 'Mean Perimeter', unit: 'mm', placeholder: '122.8' },
    { name: 'area_mean', label: 'Mean Area', unit: 'mm²', placeholder: '1001.0' },
    { name: 'smoothness_mean', label: 'Mean Smoothness', unit: '', placeholder: '0.1184' },
    { name: 'compactness_mean', label: 'Mean Compactness', unit: '', placeholder: '0.2776' },
    { name: 'concavity_mean', label: 'Mean Concavity', unit: '', placeholder: '0.3001' },
    { name: 'concave_points_mean', label: 'Mean Concave Points', unit: '', placeholder: '0.1471' },
    { name: 'symmetry_mean', label: 'Mean Symmetry', unit: '', placeholder: '0.2419' },
    { name: 'fractal_dimension_mean', label: 'Mean Fractal Dimension', unit: '', placeholder: '0.07871' }
  ],
  'Standard Error Values': [
    { name: 'radius_se', label: 'Radius Standard Error', unit: 'mm', placeholder: '1.095' },
    { name: 'texture_se', label: 'Texture Standard Error', unit: '', placeholder: '0.9053' },
    { name: 'perimeter_se', label: 'Perimeter Standard Error', unit: 'mm', placeholder: '8.589' },
    { name: 'area_se', label: 'Area Standard Error', unit: 'mm²', placeholder: '153.4' },
    { name: 'smoothness_se', label: 'Smoothness Standard Error', unit: '', placeholder: '0.006399' },
    { name: 'compactness_se', label: 'Compactness Standard Error', unit: '', placeholder: '0.04904' },
    { name: 'concavity_se', label: 'Concavity Standard Error', unit: '', placeholder: '0.05373' },
    { name: 'concave_points_se', label: 'Concave Points Standard Error', unit: '', placeholder: '0.01587' },
    { name: 'symmetry_se', label: 'Symmetry Standard Error', unit: '', placeholder: '0.03003' },
    { name: 'fractal_dimension_se', label: 'Fractal Dimension Standard Error', unit: '', placeholder: '0.006193' }
  ],
  'Worst Values': [
    { name: 'radius_worst', label: 'Worst Radius', unit: 'mm', placeholder: '25.38' },
    { name: 'texture_worst', label: 'Worst Texture', unit: '', placeholder: '17.33' },
    { name: 'perimeter_worst', label: 'Worst Perimeter', unit: 'mm', placeholder: '184.6' },
    { name: 'area_worst', label: 'Worst Area', unit: 'mm²', placeholder: '2019.0' },
    { name: 'smoothness_worst', label: 'Worst Smoothness', unit: '', placeholder: '0.1622' },
    { name: 'compactness_worst', label: 'Worst Compactness', unit: '', placeholder: '0.6656' },
    { name: 'concavity_worst', label: 'Worst Concavity', unit: '', placeholder: '0.7119' },
    { name: 'concave_points_worst', label: 'Worst Concave Points', unit: '', placeholder: '0.2654' },
    { name: 'symmetry_worst', label: 'Worst Symmetry', unit: '', placeholder: '0.4601' },
    { name: 'fractal_dimension_worst', label: 'Worst Fractal Dimension', unit: '', placeholder: '0.1189' }
  ]
};

// FeatureForm Component
const FeatureForm = ({ formData, setFormData, csvFile, setCsvFile, loading, onSubmit, onReset }) => {
  const [expandedSections, setExpandedSections] = useState({
    'Mean Values': true,
    'Standard Error Values': false,
    'Worst Values': false
  });

  const handleInputChange = (featureName, value) => {
    setFormData(prev => ({
      ...prev,
      [featureName]: value
    }));
  };

  const toggleSection = (sectionName) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionName]: !prev[sectionName]
    }));
  };

  const allFeatures = Object.values(FEATURE_GROUPS).flat();
  const hasEmptyFields = allFeatures.some(feature => !formData[feature.name] || formData[feature.name] === '');
  const filledFieldsCount = allFeatures.filter(feature => formData[feature.name] && formData[feature.name] !== '').length;

  const handleFillSampleData = () => {
    const sampleData = {};
    allFeatures.forEach(feature => {
      sampleData[feature.name] = feature.placeholder;
    });
    setFormData(sampleData);
  };

  return (
    <form onSubmit={onSubmit} className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
      {/* CSV Upload Section */}
      <div className="mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <label className="block text-gray-800 dark:text-gray-200 font-medium">Upload Patient Data (CSV):</label>
          <div className="flex items-center gap-2">
            <input 
              type="file" 
              accept=".csv" 
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) {
                  // Set the CSV file in parent component state
                  setCsvFile(file);
                }
              }} 
              disabled={loading}
              className="block text-gray-900 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 rounded p-2 border border-gray-300 dark:border-gray-600" 
            />
            {csvFile && (
              <button
                type="button"
                onClick={() => setCsvFile(null)}
                disabled={loading}
                className="px-3 py-2 text-sm bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </div>
        {csvFile && (
          <div className="mt-2 text-sm text-green-600 dark:text-green-400">
            ✓ File selected: {csvFile.name}
          </div>
        )}
        <div className="text-center text-gray-600 dark:text-gray-400 mt-2">or enter patient measurements manually below:</div>
      </div>

      {/* Form Status */}
      {!csvFile && (
        <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="flex items-center justify-between">
            <div className="text-sm text-blue-800 dark:text-blue-200">
              <span className="font-medium">Form Status:</span> {filledFieldsCount} of {allFeatures.length} fields filled
            </div>
            <button
              type="button"
              onClick={handleFillSampleData}
              disabled={loading}
              className="px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 transition-colors"
            >
              Fill Sample Data
            </button>
          </div>
          {hasEmptyFields ? (
            <div className="mt-2 text-xs text-red-600 dark:text-red-400">
              ⚠️ Please fill in all {allFeatures.length} fields to analyze patient data
            </div>
          ) : (
            <div className="mt-2 text-xs text-green-600 dark:text-green-400">
              ✅ All fields filled! Ready to analyze patient data
            </div>
          )}
        </div>
      )}

      {/* Manual Input Sections */}
      {Object.entries(FEATURE_GROUPS).map(([sectionName, features]) => (
        <div key={sectionName} className="mb-6">
          <button
            type="button"
            onClick={() => toggleSection(sectionName)}
            className="w-full flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{sectionName}</h3>
            <svg
              className={`w-5 h-5 text-gray-500 dark:text-gray-400 transform transition-transform ${
                expandedSections[sectionName] ? 'rotate-180' : ''
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {expandedSections[sectionName] && (
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {features.map((feature) => (
                <div key={feature.name} className="flex flex-col">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {feature.label}
                    {feature.unit && <span className="text-xs text-gray-500 ml-1">({feature.unit})</span>}
                  </label>
                  <input
                    type="number"
                    step="any"
                    value={formData[feature.name] || ''}
                    onChange={(e) => handleInputChange(feature.name, e.target.value)}
                    disabled={!!csvFile || loading}
                    required={!csvFile}
                    placeholder={feature.placeholder}
                    className="rounded-md p-3 bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4 justify-end pt-4 border-t border-gray-200 dark:border-gray-700">
        <button
          type="submit"
          disabled={loading || (!csvFile && hasEmptyFields)}
          className="px-8 py-4 rounded-lg bg-gradient-to-r from-green-500 to-green-600 text-white font-bold text-lg hover:from-green-600 hover:to-green-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl flex items-center justify-center min-w-[160px]"
        >
          {loading ? (
            <>
              <svg className="animate-spin h-6 w-6 mr-3 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
              </svg>
              Analyzing...
            </>
          ) : (
            <>
              <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Analyze Patient Data
            </>
          )}
        </button>
        <button
          type="button"
          onClick={onReset}
          disabled={loading}
          className="px-6 py-4 rounded-lg bg-gradient-to-r from-red-500 to-red-600 text-white font-semibold hover:from-red-600 hover:to-red-700 disabled:from-gray-400 disabled:to-gray-500 transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl"
        >
          Clear Form
        </button>
      </div>
    </form>
  );
};

// ResultsTable Component
const ResultsTable = ({ results, onViewDetails }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Batch Diagnosis Results</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          {results.length} patient{results.length !== 1 ? 's' : ''} processed
        </p>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Patient ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Diagnosis
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Confidence
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {results.map((result, index) => (
              <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                  Patient {result.sample_id !== undefined ? result.sample_id : index + 1}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    result.prediction === 'Benign' 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                      : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {result.prediction}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {(result.confidence * 100).toFixed(1)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <button
                    onClick={() => onViewDetails(result, index)}
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium transition-colors"
                  >
                    View Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// PredictionModal Component
const PredictionModal = ({ isOpen, onClose, prediction, sampleIndex, featureData }) => {
  if (!isOpen || !prediction) return null;
  // Inside PredictionModal component
  console.log('Prediction data:', prediction);
  console.log('Feature values:', prediction.feature_values);
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Patient {sampleIndex + 1} Details
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="p-6">
          {/* Prediction Summary */}
          <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Prediction Summary</h3>
                <div className="flex items-center space-x-4">
                  <span className={`text-2xl font-bold ${
                    prediction.prediction === 'Benign' 
                      ? 'text-green-600 dark:text-green-400' 
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    {prediction.prediction}
                  </span>
                  <span className="text-lg text-gray-700 dark:text-gray-300">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Benign: {(prediction.probabilities?.benign * 100 || 0).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Malignant: {(prediction.probabilities?.malignant * 100 || 0).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Feature Values */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Feature Values</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(FEATURE_GROUPS).map(([sectionName, features]) => (
                <div key={sectionName} className="space-y-3">
                  <h4 className="font-medium text-gray-700 dark:text-gray-300 text-sm uppercase tracking-wide">
                    {sectionName}
                  </h4>
                  {features.map((feature) => (
                    <div key={feature.name} className="flex justify-between items-center text-sm">
                      <span className="text-gray-600 dark:text-gray-400">{feature.label}:</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                      {prediction.feature_values?.[feature.name] ?? 'N/A'}
                        {feature.unit && <span className="text-gray-500 ml-1">{feature.unit}</span>}
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Feature Explanations (if available) */}
          {prediction.explanations && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Top Influencing Features</h3>
              <div className="space-y-2">
                {prediction.explanations.slice(0, 5).map((explanation, index) => (
                  <div key={index} className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <span className="text-gray-700 dark:text-gray-300">{explanation.feature}</span>
                    <span className={`font-medium ${
                      explanation.impact > 0 
                        ? 'text-red-600 dark:text-red-400' 
                        : 'text-green-600 dark:text-green-400'
                    }`}>
                      {explanation.impact > 0 ? '+' : ''}{explanation.impact.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-400 dark:hover:bg-gray-600 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [formData, setFormData] = useState({});
  const [csvFile, setCsvFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [dark, setDark] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [selectedSampleIndex, setSelectedSampleIndex] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError(null);
    setSuccess(false);
    setLoading(true);
    
    try {
      let response;
      
      if (csvFile) {
        // Handle CSV file upload
        const data = new FormData();
        data.append('file', csvFile);
        
        console.log('Sending CSV file to backend:', csvFile.name);
        
        response = await fetch('http://localhost:5000/predict_csv', {
          method: 'POST',
          body: data
        });
      } else {
        // Handle manual form input
        const allFeatures = Object.values(FEATURE_GROUPS).flat();
        const missingFields = allFeatures.filter(feature => !formData[feature.name] || formData[feature.name] === '');
        
        if (missingFields.length > 0) {
          const missingFieldLabels = missingFields.map(f => f.label).join(', ');
          setError(`❌ Please fill in all required fields: ${missingFieldLabels}`);
          setLoading(false);
          return;
        }

        // Convert form data to the format expected by the backend
        const json = {};
        allFeatures.forEach(feature => {
          const value = parseFloat(formData[feature.name]);
          if (isNaN(value)) {
            throw new Error(`Invalid value for ${feature.label}`);
          }
          json[feature.name] = value;
        });
        
        console.log('Sending manual data to backend:', json);
        
        response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(json)
        });
      }

      const data = await response.json();
      console.log('Backend response:', data);

      if (!response.ok) {
        console.error('Backend error:', data);
        setError(data.error || 'Prediction failed.');
        setSuccess(false);
      } else if (data.results) {
        console.log('Success! Results:', data.results);
        setResult(data.results);
        setSuccess(true);
      } else {
        console.error('Invalid response format:', data);
        setError('Invalid response from backend.');
        setSuccess(false);
      }

      setLoading(false);
    } catch (err) {
      console.error('Error during prediction:', err);
      setError(err.message || 'An unexpected error occurred.');
      setSuccess(false);
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({});
    setCsvFile(null);
    setResult(null);
    setError(null);
    setSuccess(false);
    setLoading(false);
    setModalOpen(false);
    setSelectedPrediction(null);
    setSelectedSampleIndex(null);
    
    // Also clear the file input
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleViewDetails = (prediction, index) => {
    setSelectedPrediction(prediction);
    setSelectedSampleIndex(index);
    setModalOpen(true);
  };

  return (
    <div className={dark ? 'dark min-h-screen bg-gray-900' : 'min-h-screen bg-gray-100'}>
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Breast Cancer Diagnosis Assistant</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">AI-Powered Clinical Decision Support System</p>
          </div>
          <div className="flex items-center space-x-4">
          <button
              className="rounded-lg px-4 py-2 text-sm font-medium border border-gray-500 dark:border-gray-400 text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            onClick={() => setDark(d => !d)}
            aria-label="Toggle dark mode"
          >
            {dark ? '🌙 Dark' : '☀️ Light'}
          </button>
        </div>
          </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6">
            <div className="bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-lg px-4 py-3 shadow-md border border-red-200 dark:border-red-800">
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <span className="font-bold">Error:</span>
                <span className="ml-2">{error}</span>
              </div>
            </div>
          </div>
        )}

        {/* Feature Form */}
        <FeatureForm
          formData={formData}
          setFormData={setFormData}
          csvFile={csvFile}
          setCsvFile={setCsvFile}
          loading={loading}
          onSubmit={handleSubmit}
          onReset={handleReset}
        />

        {/* Results Display */}
        {success && result && (
          <div className="mb-6">
            {result.length === 1 ? (
              // Single result display
          <div className="flex justify-center">
            <div className="w-full max-w-md bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 flex flex-col items-center">
                  <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">Diagnosis Result</h2>
                  <div className="w-full flex flex-col items-center mb-4">
                    <span className={`text-3xl font-bold mb-2 ${
                      result[0].prediction === 'Benign' 
                        ? 'text-green-600 dark:text-green-400' 
                        : 'text-red-600 dark:text-red-400'
                    }`}>
                      {result[0].prediction}
                    </span>
                    <span className="text-lg text-gray-700 dark:text-gray-200">
                      Confidence: {(result[0].confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {result[0].probabilities && (
                    <div className="w-full bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 dark:text-gray-400">Benign:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {(result[0].probabilities.benign * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-sm mt-1">
                        <span className="text-gray-600 dark:text-gray-400">Malignant:</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {(result[0].probabilities.malignant * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>
            </div>
            ) : (
              // Multiple results table
              <ResultsTable results={result} onViewDetails={handleViewDetails} />
            )}
          </div>
        )}

        {/* Prediction Modal */}
        <PredictionModal
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          prediction={selectedPrediction}
          sampleIndex={selectedSampleIndex}
          featureData={formData}
        />
      </div>
    </div>
  );
}

export default App;
