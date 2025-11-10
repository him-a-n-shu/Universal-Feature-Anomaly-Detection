import React, { useState, useMemo, useCallback } from 'react';
import { 
  UploadCloud, 
  Settings, 
  Scan, 
  Loader2, 
  AlertTriangle, 
  CheckCircle, 
  ChevronDown, 
  Image as ImageIcon,
  X,
  Database
} from 'lucide-react';

// --- Helper Components ---

/**
 * A reusable file dropzone component
 */
const FileDropzone = ({ onFilesAdded, multiple, title, subtitle }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (onFilesAdded) {
      onFilesAdded(files);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (onFilesAdded) {
      onFilesAdded(files);
    }
  };

  return (
    <div
      className={`w-full p-6 border-2 border-dashed rounded-lg text-center transition-all duration-300 ${
        isDragging ? 'border-indigo-500 bg-indigo-50' : 'border-gray-600 hover:border-indigo-400'
      }`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <input
        type="file"
        multiple={multiple}
        onChange={handleFileSelect}
        className="hidden"
        id="file-upload"
        accept="image/png, image/jpeg, image/bmp"
      />
      <label htmlFor="file-upload" className="cursor-pointer">
        <UploadCloud className="w-12 h-12 mx-auto text-gray-400" />
        <h3 className="mt-2 text-lg font-semibold text-white">{title}</h3>
        <p className="mt-1 text-sm text-gray-400">{subtitle}</p>
        <p className="mt-2 text-xs font-medium text-indigo-400">
          {isDragging ? 'Drop files here' : 'Click or drag and drop'}
        </p>
      </label>
    </div>
  );
};

/**
 * A simple loading spinner
 */
const LoadingSpinner = ({ text }) => (
  <div className="flex flex-col items-center justify-center p-8">
    <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
    <p className="mt-4 text-lg text-gray-300">{text}</p>
  </div>
);

/**
 * A component to show error messages
 */
const ErrorMessage = ({ message }) => (
  <div className="flex items-center p-4 rounded-lg bg-red-900/50 border border-red-700">
    <AlertTriangle className="w-6 h-6 mr-3 text-red-400" />
    <p className="font-medium text-red-300">{message}</p>
  </div>
);

// --- REAL API CALLS ---
// These functions now make `fetch` requests to your Python backend.

const API_BASE_URL = 'http://localhost:5001';

/**
 * REAL: Generates a product-specific coreset via the backend.
 * This POSTs to the `/setup` endpoint.
 */
const generateCoresetAPI = async (productName, goldenImages) => {
  console.log(`[API CALL] Generating coreset for '${productName}' with ${goldenImages.length} images.`);
  
  const formData = new FormData();
  formData.append('product_name', productName);
  goldenImages.forEach((file) => {
    formData.append('files', file);
  });

  const response = await fetch(`${API_BASE_URL}/setup`, {
    method: 'POST',
    body: formData,
    // Note: Don't set 'Content-Type' for FormData, browser does it automatically with boundary.
  });

  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || 'Failed to generate coreset.');
  }
  return result; // e.g., { success: true, message: "...", ... }
};

/**
 * REAL: Runs inference on a test image via the backend.
 * This POSTs to the `/inference` endpoint.
 */
const runInferenceAPI = async (productName, testImage) => {
  console.log(`[API CALL] Running inference for '${productName}' on image '${testImage.name}'.`);
  
  const formData = new FormData();
  formData.append('product_name', productName);
  formData.append('file', testImage);

  const response = await fetch(`${API_BASE_URL}/inference`, {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || 'Failed to run inference.');
  }
  return result; // e.g., { decision, score, original_image_b64, heatmap_image_b64 }
};


// --- Main Application Views ---

/**
 * View 1: Product Setup (Fine-tuning Coreset Generation)
 */
const SetupView = ({ coresetStatus, onCoresetGenerated }) => {
  const [productName, setProductName] = useState('');
  const [goldenImages, setGoldenImages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleFilesAdded = (files) => {
    setGoldenImages(files);
    setError(null);
    setSuccess(null);
  };

  const handleRemoveImage = (index) => {
    setGoldenImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleGenerateCoreset = async () => {
    if (!productName || goldenImages.length === 0) {
      setError('Please provide a product name and upload at least one golden image.');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await generateCoresetAPI(productName, goldenImages);
      setSuccess(result.message);
      onCoresetGenerated(productName); // Update parent state
      // Clear form on success
      setProductName('');
      setGoldenImages([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const imagePreviews = useMemo(() => {
    return goldenImages.map((file, index) => ({
      name: file.name,
      url: URL.createObjectURL(file),
      index: index
    }));
  }, [goldenImages]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 p-8">
      {/* Left Column: Input Form */}
      <div className="lg:col-span-2 bg-gray-800 p-8 rounded-xl shadow-2xl">
        <h2 className="text-3xl font-bold text-white mb-6">Create New Product Coreset</h2>
        
        {error && <ErrorMessage message={error} />}
        {success && (
          <div className="flex items-center p-4 rounded-lg bg-green-900/50 border border-green-700 mb-4">
            <CheckCircle className="w-6 h-6 mr-3 text-green-400" />
            <p className="font-medium text-green-300">{success}</p>
          </div>
        )}

        <div className="space-y-6">
          <div>
            <label htmlFor="product-name" className="block text-sm font-medium text-gray-300 mb-2">
              Product Name
            </label>
            <input
              type="text"
              id="product-name"
              value={productName}
              onChange={(e) => setProductName(e.target.value)}
              placeholder="e.g., 'Carpet', 'Bottle', 'PCB'"
              className="w-full bg-gray-700 text-white border-gray-600 rounded-lg px-4 py-3 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Upload Golden Samples (Good Images)
            </label>
            <FileDropzone
              onFilesAdded={handleFilesAdded}
              multiple={true}
              title="Upload Good Product Images"
              subtitle={`Drop ${goldenImages.length > 0 ? `${goldenImages.length} files selected` : 'files here'}`}
            />
          </div>

          {imagePreviews.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">Selected Images:</h4>
              <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto p-2 bg-gray-900 rounded-lg">
                {imagePreviews.map(img => (
                  <div key={img.index} className="relative w-16 h-16">
                    <img src={img.url} alt={img.name} className="w-16 h-16 object-cover rounded-md" />
                    <button 
                      onClick={() => handleRemoveImage(img.index)}
                      className="absolute -top-1 -right-1 bg-red-600 text-white rounded-full p-0.5"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <button
            onClick={handleGenerateCoreset}
            disabled={isLoading}
            className="w-full flex items-center justify-center bg-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-indigo-700 disabled:bg-gray-500 transition-all"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              'Generate Product Coreset'
            )}
          </button>
        </div>
      </div>

      {/* Right Column: Status */}
      <div className="bg-gray-800 p-8 rounded-xl shadow-2xl">
        <h3 className="text-2xl font-bold text-white mb-6">Product Status</h3>
        <div className="space-y-4">
          <div className="flex items-center text-gray-400">
            <Database className="w-5 h-5 mr-3 text-indigo-400" />
            <span>Coresets Available: {coresetStatus.size}</span>
          </div>
          {coresetStatus.size === 0 ? (
            <p className="text-sm text-gray-500">No product coresets have been generated yet. Use the form to create one.</p>
          ) : (
            <ul className="space-y-2 max-h-80 overflow-y-auto">
              {[...coresetStatus].sort().map(product => (
                <li key={product} className="flex items-center p-3 bg-gray-700 rounded-lg">
                  <CheckCircle className="w-5 h-5 mr-3 text-green-400" />
                  <span className="font-medium text-gray-200 capitalize">{product}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * View 2: Live Inference (Anomaly Detection)
 */
const InferenceView = ({ coresetStatus }) => {
  const [selectedProduct, setSelectedProduct] = useState('');
  const [testImage, setTestImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Auto-select the first product if one becomes available
  React.useEffect(() => {
    if (!selectedProduct && coresetStatus.size > 0) {
      setSelectedProduct([...coresetStatus][0]);
    }
  }, [coresetStatus, selectedProduct]);

  const handleFileAdded = (files) => {
    if (files.length > 0) {
      setTestImage(files[0]);
      setResult(null); // Clear previous result
      setError(null);
    }
  };

  const handleRunInference = async () => {
    if (!selectedProduct || !testImage) {
      setError('Please select a product and upload a test image.');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiResult = await runInferenceAPI(selectedProduct, testImage);
      setResult(apiResult);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const testImageUrl = useMemo(() => {
    return testImage ? URL.createObjectURL(testImage) : null;
  }, [testImage]);

  return (
    <div className="p-8">
      {/* Top Section: Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-gray-800 p-8 rounded-xl shadow-2xl">
          <h2 className="text-3xl font-bold text-white mb-6">Run Anomaly Detection</h2>
          
          {error && <div className="mb-4"><ErrorMessage message={error} /></div>}

          <div className="space-y-6">
            <div>
              <label htmlFor="product-select" className="block text-sm font-medium text-gray-300 mb-2">
                1. Select Product
              </label>
              {coresetStatus.size === 0 ? (
                   <p className="text-sm text-yellow-400 p-3 bg-yellow-900/50 rounded-lg border border-yellow-700">
                      No products found. Please go to "Product Setup" to create a coreset first.
                   </p>
               ) : (
                 <div className="relative">
                   <select
                     id="product-select"
                     value={selectedProduct}
                     onChange={(e) => setSelectedProduct(e.target.value)}
                     className="w-full bg-gray-700 text-white border-gray-600 rounded-lg px-4 py-3 appearance-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                   >
                     <option value="" disabled>-- Select a product --</option>
                     {[...coresetStatus].sort().map(product => (
                         <option key={product} value={product} className="capitalize">{product}</option>
                     ))}
                   </select>
                   <ChevronDown className="w-5 h-5 text-gray-400 absolute right-4 top-3.5" />
                 </div>
               )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                2. Upload Test Image
              </label>
              {testImage ? (
                <div className="flex items-center gap-4">
                  <img src={testImageUrl} alt="Test preview" className="w-24 h-24 object-cover rounded-lg" />
                  <div>
                    <p className="text-sm font-medium text-white">{testImage.name}</p>
                    <button 
                      onClick={() => setTestImage(null)}
                      className="text-sm text-red-400 hover:text-red-300"
                    >
                      Remove Image
                    </button>
                  </div>
                </div>
              ) : (
                <FileDropzone
                  onFilesAdded={handleFileAdded}
                  multiple={false}
                  title="Upload Test Image"
                  subtitle="Drop a single image here"
                />
              )}
            </div>
            
            <button
              onClick={handleRunInference}
              disabled={isLoading || !testImage || !selectedProduct}
              className="w-full flex items-center justify-center bg-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:bg-indigo-700 disabled:bg-gray-500 transition-all"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Running Inference...
                </>
              ) : (
                'Run Detection'
              )}
            </button>
          </div>
        </div>
        
        {/* Placeholder for when no result is ready */}
        {!isLoading && !result && (
           <div className="bg-gray-800 p-8 rounded-xl shadow-2xl flex flex-col items-center justify-center text-center">
             <ImageIcon className="w-24 h-24 text-gray-600" />
             <h3 className="mt-4 text-xl font-semibold text-gray-400">Results will appear here</h3>
             <p className="mt-2 text-sm text-gray-500">
               Select a product, upload an image, and click "Run Detection" to see the anomaly analysis.
             </p>
           </div>
        )}
        
        {/* Loading State */}
        {isLoading && (
          <div className="bg-gray-800 p-8 rounded-xl shadow-2xl">
            <LoadingSpinner text="Analyzing image... this may take a moment." />
          </div>
        )}
        
        {/* Result Display */}
        {result && !isLoading && (
          <div className="bg-gray-800 p-8 rounded-xl shadow-2xl">
            <h2 className="text-3xl font-bold text-white mb-6">Detection Result</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Decision */}
              <div className={`p-6 rounded-lg ${
                result.decision.includes('ANOMALOUS') ? 'bg-red-900/50' : 'bg-green-900/50'
              }`}>
                <h4 className="text-sm font-semibold text-gray-400 mb-1">Decision</h4>
                <p className={`text-3xl font-bold ${
                  result.decision.includes('ANOMALOUS') ? 'text-red-300' : 'text-green-300'
                }`}>
                  {result.decision}
                </p>
              </div>
              {/* Score */}
              <div className="bg-gray-700 p-6 rounded-lg">
                <h4 className="text-sm font-semibold text-gray-400 mb-1">Anomaly Score</h4>
                <p className="text-3xl font-bold text-white">{result.score}</p>
                <p className="text-xs text-gray-500">
                  (Higher score = higher chance of anomaly)
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Section: Image Results */}
      {result && !isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-gray-800 p-6 rounded-xl shadow-2xl">
            <h3 className="text-xl font-semibold text-white mb-4">Original Image</h3>
            <img 
              src={result.original_image_b64} 
              alt="Original Test" 
              className="w-full h-auto rounded-lg object-contain max-h-96" 
            />
          </div>
          <div className="bg-gray-800 p-6 rounded-xl shadow-2xl">
            <h3 className="text-xl font-semibold text-white mb-4">Anomaly Heatmap</h3>
            <img 
              src={result.heatmap_image_b64} 
              alt="Anomaly Heatmap" 
              className="w-full h-auto rounded-lg object-contain max-h-96" 
            />
          </div>
        </div>
      )}

    </div>
  );
};


/**
 * Main App Component
 */
export default function App() {
  const [view, setView] = useState('setup'); // 'setup' or 'inference'
  
  // This state holds which products have a coreset.
  // We'll pre-populate it for the demo.
  const [coresetStatus, setCoresetStatus] = useState(new Set(['carpet']));

  const handleCoresetGenerated = (productName) => {
    setCoresetStatus(prevStatus => {
      const newStatus = new Set(prevStatus);
      newStatus.add(productName);
      return newStatus;
    });
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 font-inter">
      {/* Header & Navigation */}
      <header className="bg-gray-800 shadow-lg">
        {/* THIS IS THE FIX:
          The 'container' and 'mx-auto' classes have been removed from this <nav> element.
          It now only has 'px-8' for padding.
        */}
        <nav className="px-8 flex items-center justify-between h-20">
          <h1 className="text-2xl font-bold text-white">
            Industrial Anomaly Detection
          </h1>
          <div className="flex items-center space-x-2 bg-gray-900 rounded-lg p-1">
            <button
              onClick={() => setView('setup')}
              className={`flex items-center px-4 py-2 rounded-md font-medium transition-all ${
                view === 'setup'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:bg-gray-700'
              }`}
            >
              <Settings className="w-5 h-5 mr-2" />
              Product Setup
            </button>
            <button
              onClick={() => setView('inference')}
              className={`flex items-center px-4 py-2 rounded-md font-medium transition-all ${
                view === 'inference'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:bg-gray-700'
              }`}
            >
              <Scan className="w-5 h-5 mr-2" />
              Live Inference
            </button>
          </div>
        </nav>
      </header>

      {/* THIS IS THE SECOND PART OF THE FIX:
        The 'container' and 'mx-auto' classes have been removed from this <main> element.
        It now only has 'px-8' for padding.
      */}
      <main className="px-8">
        {view === 'setup' ? (
          <SetupView 
            coresetStatus={coresetStatus} 
            onCoresetGenerated={handleCoresetGenerated} 
          />
        ) : (
          <InferenceView coresetStatus={coresetStatus} />
        )}
      </main>
      
      <footer className="text-center p-8 text-gray-600 text-sm">
        UI built based on `fine-tuning-ad.ipynb` and `inference.ipynb` workflows.
      </footer>
    </div>
  );
}