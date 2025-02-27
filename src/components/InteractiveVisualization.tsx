import React, { useState } from 'react';
import { Sliders } from 'lucide-react';

interface Parameter {
  name: string;
  min: number;
  max: number;
  step: number;
  value: number;
}

export function InteractiveVisualization() {
  const [parameters, setParameters] = useState<Parameter[]>([
    { name: 'Learning Rate', min: 0.001, max: 1, step: 0.001, value: 0.01 },
    { name: 'Epochs', min: 1, max: 100, step: 1, value: 10 },
  ]);

  const handleParameterChange = (index: number, newValue: number) => {
    setParameters(prev => 
      prev.map((param, i) => 
        i === index ? { ...param, value: newValue } : param
      )
    );
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-6">
        <Sliders className="text-indigo-600" size={24} />
        <h3 className="text-xl font-semibold">Model Parameters</h3>
      </div>

      <div className="space-y-6">
        {parameters.map((param, index) => (
          <div key={param.name} className="space-y-2">
            <div className="flex justify-between">
              <label className="text-sm font-medium text-gray-700">
                {param.name}
              </label>
              <span className="text-sm text-gray-500">
                {param.value}
              </span>
            </div>
            <input
              type="range"
              min={param.min}
              max={param.max}
              step={param.step}
              value={param.value}
              onChange={(e) => handleParameterChange(index, parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
          </div>
        ))}
      </div>

      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Model Output</h4>
        <div className="h-64 bg-white rounded border border-gray-200">
          {/* Placeholder for visualization */}
          <div className="flex items-center justify-center h-full text-gray-400">
            Visualization will appear here
          </div>
        </div>
      </div>
    </div>
  );
}