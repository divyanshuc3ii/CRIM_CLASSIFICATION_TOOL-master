import React, { useState } from 'react';
import axios from 'axios';

function CrimeClassificationApp() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/api/predict/', { text: inputText }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      setResult(response.data);
    } catch (err) {
      setError('Failed to classify the text. Please try again.');
      console.error('Classification error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4 text-center">Crime Classification</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter text to classify..."
          className="w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows="4"
          required
        />
        <button 
          type="submit" 
          disabled={loading}
          className="w-full bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 transition duration-300 disabled:opacity-50"
        >
          {loading ? 'Classifying...' : 'Classify'}
        </button>
      </form>

      {error && (
        <div className="mt-4 text-red-500 text-center">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded-md">
          <h3 className="font-bold mb-2">Classification Result:</h3>
          <pre className="whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default CrimeClassificationApp;