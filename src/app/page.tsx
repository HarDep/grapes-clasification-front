'use client';
import React, { useState, useRef, useCallback } from 'react';
import { Upload, Camera, RotateCcw, Info, AlertTriangle, CheckCircle } from 'lucide-react';
import { motion} from 'framer-motion';

// Tipos de datos
interface VerificationResponse {
  is_grape_leaf: boolean;
  grape_probability: number;
  message: string;
}

interface ClassificationResponse {
  predicted_class: string;
  confidence: number;
  all_predictions: Record<string, number>;
  disease_info: {
    emoji: string;
    description: string;
    severity: string;
    treatment: string;
  };
}

interface ProcessingStep {
  name: string;
  description: string;
  duration: number;
}

const GrapeDiseaseClassifier = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerificationResponse | null>(null);
  const [classificationResult, setClassificationResult] = useState<ClassificationResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const timeoutRefs = useRef<NodeJS.Timeout[]>([]);

  // Pasos del proceso de clasificación
  const processingSteps: ProcessingStep[] = [
    { name: "Preprocesamiento", description: "Redimensionando a 128x128 píxeles", duration: 4000 },
    { name: "Capa Conv2D #1", description: "Aplicando filtros convolucionales", duration: 4000 },
    { name: "MaxPooling #1", description: "Reduciendo dimensionalidad", duration: 4000 },
    { name: "Capa Conv2D #2", description: "Extrayendo características avanzadas", duration: 4000 },
    { name: "MaxPooling #2", description: "Comprimiendo información", duration: 4000 },
    { name: "Capa Conv2D #3", description: "Características de alto nivel", duration: 4000 },
    { name: "MaxPooling #3", description: "Reducción final", duration: 4000 },
    { name: "Flatten", description: "Convirtiendo a vector 1D", duration: 4000 },
    { name: "Capa Densa", description: "Procesamiento neuronal denso", duration: 4000 },
    { name: "Dropout", description: "Regulación de sobreajuste", duration: 4000 }
  ];

  // Información de enfermedades
  const diseaseInfo = {
    "Black Rot": {
      emoji: "🔴",
      description: "Podredumbre negra - Enfermedad fúngica grave que causa manchas circulares marrones en las hojas.",
      severity: "Alta",
      treatment: "Fungicidas cúpricos, poda de partes afectadas, mejora de ventilación"
    },
    "ESCA (Black measles)": {
      emoji: "🟤",
      description: "Enfermedad de la ESCA - Complejo de hongos que causa manchas como sarampión en las hojas.",
      severity: "Muy Alta",
      treatment: "No hay cura definitiva, manejo preventivo con fungicidas sistémicos"
    },
    "Healthy": {
      emoji: "🟢",
      description: "Hoja saludable - Sin signos de enfermedad detectables.",
      severity: "Ninguna",
      treatment: "Continuar con prácticas de manejo preventivo"
    },
    "Leaf Blight": {
      emoji: "🟡",
      description: "Tizón de la hoja - Enfermedad que causa manchas necróticas y amarillamiento.",
      severity: "Media",
      treatment: "Fungicidas preventivos, mejora del drenaje, evitar riego foliar"
    }
  };

  const handleImageSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/') && (file.type == 'image/jpeg' || file.type == 'image/jpeg')) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setVerificationResult(null);
      setClassificationResult(null);
      setCurrentStep(0);
      resetSimulation();
    }
  }, []);

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setVerificationResult(null);
      setClassificationResult(null);
      setCurrentStep(0);
      resetSimulation();
    }
  }, []);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  }, []);

  const resetSimulation = () => {
    setIsProcessing(false);
    setCurrentStep(-1);
    setIsPaused(false);
    timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
    timeoutRefs.current = [];
    
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx!.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const applyChannelFilter = (imageData: ImageData, channel: 'red' | 'green' | 'blue') => {
    const data = imageData.data;
    const filteredData = new Uint8ClampedArray(data);
    
    for (let i = 0; i < data.length; i += 4) {
      switch (channel) {
        case 'red':
          filteredData[i + 1] = data[i + 1] * 0.3; // Reducir green
          filteredData[i + 2] = data[i + 2] * 0.3; // Reducir blue
          break;
        case 'green':
          filteredData[i] = data[i] * 0.3; // Reducir red
          filteredData[i + 2] = data[i + 2] * 0.3; // Reducir blue
          break;
        case 'blue':
          filteredData[i] = data[i] * 0.3; // Reducir red
          filteredData[i + 1] = data[i + 1] * 0.3; // Reducir green
          break;
        default:
          break;
      }
    }
    
    return new ImageData(filteredData, imageData.width, imageData.height);
  };

  const downsampleImage = (imageData: ImageData, targetWidth: number, targetHeight: number) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Canvas temporal con la imagen original
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCtx!.putImageData(imageData, 0, 0);
    
    // Redimensionar
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    ctx!.drawImage(tempCanvas, 0, 0, imageData.width, imageData.height, 0, 0, targetWidth, targetHeight);
    
    return ctx!.getImageData(0, 0, targetWidth, targetHeight);
  };

  const drawNeuralNetwork = (ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number) => {
    ctx.strokeStyle = '#60a5fa';
    ctx.fillStyle = '#60a5fa';
    ctx.lineWidth = 2;
    
    // Dibujar nodos
    const nodeCount = 8;
    const nodeSpacing = height / (nodeCount + 1);
    
    for (let i = 0; i < nodeCount; i++) {
      const nodeY = y + (i + 1) * nodeSpacing;
      
      // Nodo de entrada
      ctx.beginPath();
      ctx.arc(x + 20, nodeY, 6, 0, Math.PI * 2);
      ctx.fill();
      
      // Nodo de salida
      ctx.beginPath();
      ctx.arc(x + width - 20, nodeY, 6, 0, Math.PI * 2);
      ctx.fill();
      
      // Conexión
      ctx.beginPath();
      ctx.moveTo(x + 26, nodeY);
      ctx.lineTo(x + width - 26, nodeY);
      ctx.stroke();
    }
  };

  const simulateImageProcessing = async () => {
    if (!imagePreview) return;
    
    // Asegurar que el canvas esté disponible
    if (!canvasRef.current) {
      // Esperar un momento y reintentar
      setTimeout(() => {
        simulateImageProcessing();
      }, 100);
      return;
    }
    
    setIsProcessing(true);
    setCurrentStep(0);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = 400;
      canvas.height = 300;
      
      let currentImageData: ImageData | null = null;
      let currentWidth = 128;
      let currentHeight = 128;
      
      const executeStep = (step: number) => {
        if (!ctx || isPaused) return;
        
        setCurrentStep(step);
        
        // Limpiar canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Fondo
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Título del paso
        ctx.fillStyle = '#f1f5f9';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(processingSteps[step].name, canvas.width / 2, 30);
        
        // Descripción
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Arial';
        ctx.fillText(processingSteps[step].description, canvas.width / 2, 50);
        
        // Área de imagen centrada
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2 + 10;
        
        switch (step) {
          case 0: // Preprocesamiento
            ctx.drawImage(img, centerX - 64, centerY - 64, 128, 128);
            currentImageData = ctx.getImageData(centerX - 64, centerY - 64, 128, 128);
            currentWidth = 128;
            currentHeight = 128;
            break;
            
          case 1: // Conv2D #1 - Red
            if (currentImageData) {
              const redFiltered = applyChannelFilter(currentImageData, 'red');
              ctx.putImageData(redFiltered, centerX - currentWidth/2, centerY - currentHeight/2);
              currentImageData = redFiltered;
            }
            break;
            
          case 2: // MaxPooling #1
            if (currentImageData) {
              const downsampled = downsampleImage(currentImageData, 64, 64);
              ctx.putImageData(downsampled, centerX - 32, centerY - 32);
              currentImageData = downsampled;
              currentWidth = 64;
              currentHeight = 64;
            }
            break;
            
          case 3: // Conv2D #2 - Green
            if (currentImageData) {
              const greenFiltered = applyChannelFilter(currentImageData, 'green');
              ctx.putImageData(greenFiltered, centerX - currentWidth/2, centerY - currentHeight/2);
              currentImageData = greenFiltered;
            }
            break;
            
          case 4: // MaxPooling #2
            if (currentImageData) {
              const downsampled = downsampleImage(currentImageData, 32, 32);
              ctx.putImageData(downsampled, centerX - 16, centerY - 16);
              currentImageData = downsampled;
              currentWidth = 32;
              currentHeight = 32;
            }
            break;
            
          case 5: // Conv2D #3 - Blue
            if (currentImageData) {
              const blueFiltered = applyChannelFilter(currentImageData, 'blue');
              ctx.putImageData(blueFiltered, centerX - currentWidth/2, centerY - currentHeight/2);
              currentImageData = blueFiltered;
            }
            break;
            
          case 6: // MaxPooling #3
            if (currentImageData) {
              const downsampled = downsampleImage(currentImageData, 16, 16);
              ctx.putImageData(downsampled, centerX - 8, centerY - 8);
              currentImageData = downsampled;
              currentWidth = 16;
              currentHeight = 16;
            }
            break;
            
          case 7: // Flatten
            // Mostrar vector como líneas
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            const vectorLength = 200;
            const vectorStart = centerX - vectorLength/2;
            
            for (let i = 0; i < 20; i++) {
              const x = vectorStart + (i * vectorLength/20);
              const height = Math.random() * 40 + 10;
              ctx.beginPath();
              ctx.moveTo(x, centerY + 20);
              ctx.lineTo(x, centerY + 20 - height);
              ctx.stroke();
            }
            break;
            
          case 8: // Capa Densa
            drawNeuralNetwork(ctx, centerX - 100, centerY - 50, 200, 100);
            break;
            
          case 9: // Dropout
            drawNeuralNetwork(ctx, centerX - 100, centerY - 50, 200, 100);
            // Añadir efecto de dropout
            ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
            for (let i = 0; i < 3; i++) {
              const nodeY = centerY - 30 + (i * 30);
              ctx.beginPath();
              ctx.arc(centerX - 80 + Math.random() * 160, nodeY, 8, 0, Math.PI * 2);
              ctx.fill();
            }
            break;
        }
        
        // Programar siguiente paso
        if (step < processingSteps.length - 1) {
          const timeout = setTimeout(() => {
            executeStep(step + 1);
          }, processingSteps[step].duration);
          timeoutRefs.current.push(timeout);
        } else {
          setIsProcessing(false);
          setCurrentStep(-1);
        }
      };
      
      executeStep(0);
    };
    
    img.src = imagePreview;
  };

  const processImage = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setCurrentStep(0);
    setIsPaused(true);

    try {
      
      // Paso 1: Verificación
      setCurrentStep(1);
      const verifyFormData = new FormData();
      verifyFormData.append('image', selectedImage);

      const verifyResponse = await fetch('https://grape-leaf-verificator.onrender.com/predict', {
        method: 'POST',
        body: verifyFormData,
      });

      if (!verifyResponse.ok) {
        throw new Error('Error en la verificación');
      }

      const verifyData: VerificationResponse = await verifyResponse.json();
      setVerificationResult(verifyData);
      
      if (verifyData.is_grape_leaf) {
        // Clasificación de enfermedad
        const classifyFormData = new FormData();
        classifyFormData.append('image', selectedImage);

        const classifyResponse = await fetch('https://grape-disease-classifier.onrender.com/predict', {
          method: 'POST',
          body: classifyFormData,
        });

        if (!classifyResponse.ok) {
          throw new Error('Error en la clasificación');
        }

        const classifyData: ClassificationResponse = await classifyResponse.json();

        let is_grape_leaf = false;

        for (const key in classifyData.all_predictions) {
          const pred = classifyData.all_predictions[key];
          if (pred > 0.5) {
            is_grape_leaf = true;
            break;
          }
        }

        if (!is_grape_leaf) {
          setVerificationResult({
            is_grape_leaf: false,
            grape_probability: 0,
            message: '❌ La imagen no parece ser una hoja de uva'
          })
          return;
        }

        // Simular el proceso visual
        simulateImageProcessing();
        // Simular pasos de procesamiento
        for (let i = 2; i <= processingSteps.length; i++) {
          await new Promise(resolve => setTimeout(resolve, processingSteps[i-1]?.duration || 1000));
          setCurrentStep(i);
        }

        setClassificationResult(classifyData);
      }

    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error procesando la imagen: ' + (error as Error).message);
    } finally {
      setIsProcessing(false);
      setIsPaused(false);
    }
  };

  const resetAll = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setVerificationResult(null);
    setClassificationResult(null);
    setCurrentStep(0);
    setIsPaused(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div 
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl font-bold text-green-800 mb-4 flex items-center justify-center gap-3">
            🍇 Clasificador Inteligente de Enfermedades en Hojas de Uva
          </h1>
          <p className="text-lg text-green-700 max-w-3xl mx-auto">
            Esta aplicación utiliza <strong>Inteligencia Artificial</strong> con redes neuronales convolucionales 
            para detectar enfermedades en hojas de uva de manera rápida y precisa.
          </p>
        </motion.div>

        {/* Disease Info Cards */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {Object.entries(diseaseInfo).map(([disease, info]) => (
            <div key={disease} className="bg-white p-4 rounded-lg shadow-md border-l-4 border-green-500">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl">{info.emoji}</span>
                <span className="font-semibold text-sm">{disease}</span>
              </div>
              <p className="text-xs text-gray-600 mb-1">{info.description}</p>
              <span className={`text-xs px-2 py-1 rounded-full ${
                info.severity === 'Ninguna' ? 'bg-green-100 text-green-800' :
                info.severity === 'Media' ? 'bg-yellow-100 text-yellow-800' :
                info.severity === 'Alta' ? 'bg-orange-100 text-orange-800' :
                'bg-red-100 text-red-800'
              }`}>
                {info.severity}
              </span>
            </div>
          ))}
        </motion.div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Upload */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                📸 Subir Imagen de Hoja de Uva
              </h2>
              
              {/* Image Upload Area */}
              <div
                className={`border-2 border-dashed border-gray-300 rounded-lg p-8 text-center transition-colors duration-200 ${
                  !selectedImage ? 'hover:border-green-400 hover:bg-green-50' : ''
                }`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                {imagePreview ? (
                  <div className="space-y-4">
                    <img
                      src={imagePreview}
                      alt="Imagen seleccionada"
                      className="max-w-full max-h-64 mx-auto rounded-lg shadow-md"
                    />
                    <div className="flex gap-2 justify-center">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2"
                      >
                        <Upload size={16} />
                        Cambiar Imagen
                      </button>
                      <button
                        onClick={resetAll}
                        className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2"
                      >
                        <RotateCcw size={16} />
                        Limpiar
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Camera size={48} className="mx-auto text-gray-400" />
                    <div>
                      <p className="text-lg font-medium text-gray-700">
                        Arrastra una imagen aquí o haz clic para seleccionar
                      </p>
                      <p className="text-sm text-gray-500 mt-2">
                        Formatos soportados: JPG, JPEG
                      </p>
                    </div>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-6 py-3 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors flex items-center gap-2 mx-auto"
                    >
                      <Upload size={20} />
                      Seleccionar Imagen
                    </button>
                  </div>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />

              {/* Analyze Button */}
              {selectedImage && (
                <motion.button
                  onClick={processImage}
                  disabled={isProcessing}
                  className={`w-full mt-6 py-4 px-6 rounded-lg font-semibold text-lg transition-all duration-200 flex items-center justify-center gap-3 ${
                    isProcessing
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 shadow-lg hover:shadow-xl'
                  } text-white`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  whileHover={{ scale: isProcessing ? 1 : 1.02 }}
                  whileTap={{ scale: isProcessing ? 1 : 0.98 }}
                >
                  {isProcessing ? (
                    <>
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                      Analizando...
                    </>
                  ) : (
                    <>
                      🔍 Analizar Imagen
                    </>
                  )}
                </motion.button>
              )}
            </div>
          </motion.div>

          {/* Right Column - Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                📋 Resultado del Análisis
              </h2>

              {!selectedImage && (
                <div className="text-center py-12 text-gray-500">
                  <Info size={48} className="mx-auto mb-4 text-gray-300" />
                  <p>👆 Sube una imagen para comenzar el análisis</p>
                </div>
              )}

              {/* Video Visualization */}
              {(isProcessing || verificationResult) && (
                <div className="mb-6">
                  <div className="bg-gray-900 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-white font-medium">🎥 Proceso de Clasificación</h3>
                    </div>
                    <div className="bg-black rounded flex items-center justify-center" style={{ height: '300px' }}>
                      <canvas
                        ref={canvasRef}
                        width={128 + 128}
                        height={128 + 128}
                        className="border border-gray-600 rounded"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Processing Steps */}
              {isProcessing && (
                <div className="space-y-3 mb-6">
                  {processingSteps.map((step, index) => (
                    <motion.div
                      key={index}
                      className={`flex items-center gap-3 p-3 rounded-lg transition-all duration-300 ${
                        index < currentStep
                          ? 'bg-green-100 text-green-800'
                          : index === currentStep
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-gray-100 text-gray-500'
                      }`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold ${
                        index < currentStep
                          ? 'bg-green-500 text-white'
                          : index === currentStep
                          ? 'bg-blue-500 text-white animate-pulse'
                          : 'bg-gray-300 text-gray-600'
                      }`}>
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-semibold">{step.name}</div>
                        <div className="text-sm opacity-75">{step.description}</div>
                      </div>
                      {index === currentStep && (
                        <div className="ml-auto">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}

              {/* Verification Result */}
              {verificationResult && !verificationResult.is_grape_leaf && (
                <motion.div
                  className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <div className="flex items-center gap-3 mb-3">
                    <AlertTriangle className="text-red-500" size={24} />
                    <h3 className="font-semibold text-red-800">❌ Imagen No Válida</h3>
                  </div>
                  <div className="space-y-2 text-sm">
                    <p><strong>Estado:</strong> No es una hoja de uva</p>
                    <p className="text-red-700">{verificationResult.message}</p>
                  </div>
                  <div className="mt-4 p-3 bg-red-100 rounded">
                    <h4 className="font-medium text-red-800 mb-2">📋 Instrucciones</h4>
                    <ul className="text-sm text-red-700 space-y-1">
                      <li>✅ Una hoja de uva clara y visible</li>
                      <li>✅ Buena iluminación</li>
                      <li>✅ Enfoque adecuado</li>
                      <li>✅ Sin objetos que obstruyan la vista</li>
                    </ul>
                  </div>
                </motion.div>
              )}

              {/* Classification Result */}
              {verificationResult?.is_grape_leaf && classificationResult && (
                <motion.div
                  className="space-y-4"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  {/* Success validation */}
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-2">
                      <CheckCircle className="text-green-500" size={20} />
                      <span className="font-semibold text-green-800">✅ Validación Exitosa</span>
                    </div>
                    <p className="text-sm text-green-700">
                      Imagen validada como hoja de uva
                    </p>
                  </div>

                  {/* Main diagnosis */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h3 className="font-semibold text-blue-800 mb-3 flex items-center gap-2">
                      {diseaseInfo[classificationResult.predicted_class as keyof typeof diseaseInfo]?.emoji} 
                      Diagnóstico Principal
                    </h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Clase Detectada:</span>
                        <p className="text-blue-700">{classificationResult.predicted_class}</p>
                      </div>
                      <div>
                        <span className="font-medium">Confianza:</span>
                        <p className="text-blue-700">{(classificationResult.confidence * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                  </div>

                  {/* Probabilities */}
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <h4 className="font-semibold mb-3">📊 Probabilidades Detalladas</h4>
                    <div className="space-y-2">
                      {Object.entries(classificationResult.all_predictions)
                        .sort(([,a], [,b]) => b - a)
                        .map(([disease, probability]) => (
                          <div key={disease} className="flex items-center justify-between">
                            <span className="flex items-center gap-2 text-sm">
                              {diseaseInfo[disease as keyof typeof diseaseInfo]?.emoji}
                              {disease}
                            </span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                                  style={{ width: `${probability * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-sm font-medium w-12 text-right">
                                {(probability * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>

                  {/* Disease info and treatment */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <h4 className="font-semibold text-yellow-800 mb-2">📋 Información de la Enfermedad</h4>
                      <p className="text-sm text-yellow-700">
                        {classificationResult.disease_info.description}
                      </p>
                    </div>
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                      <h4 className="font-semibold text-purple-800 mb-2">💊 Recomendaciones</h4>
                      <p className="text-sm text-purple-700">
                        {classificationResult.disease_info.treatment}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default GrapeDiseaseClassifier;