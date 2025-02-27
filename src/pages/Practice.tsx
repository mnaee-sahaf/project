import React, { useState, useEffect } from 'react';
import { Dumbbell, Search, Brain, ArrowRight, BarChart2 } from 'lucide-react';
import { InteractiveVisualization } from '../components/InteractiveVisualization';
import { supabase } from '../lib/supabase';
import type { Exercise } from '../types';

export function Practice() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [activeExercise, setActiveExercise] = useState<Exercise | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exercises, setExercises] = useState<Exercise[]>([]);

  useEffect(() => {
    const fetchExercises = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        
        if (!user) {
          setError('Please sign in to access exercises');
          return;
        }

        // In a real app, we would fetch this from the database
        // For now, using static data
        setExercises([
          {
            id: 'ex1',
            lessonId: '2',
            title: 'Linear Regression Parameter Tuning',
            description: 'Experiment with learning rate and epochs to optimize a linear regression model.',
            type: 'interactive',
            completed: false,
          },
          {
            id: 'ex2',
            lessonId: '3',
            title: 'K-NN Classification Boundaries',
            description: 'Visualize how different k values affect decision boundaries in K-Nearest Neighbors.',
            type: 'visualization',
            completed: false,
          },
          {
            id: 'ex3',
            lessonId: '1',
            title: 'ML Fundamentals Quiz',
            description: 'Test your knowledge of basic machine learning concepts and terminology.',
            type: 'quiz',
            completed: false,
          },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load exercises');
      } finally {
        setLoading(false);
      }
    };

    fetchExercises();
  }, []);

  const filteredExercises = exercises.filter(exercise => {
    const matchesSearch = exercise.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         exercise.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = !selectedType || exercise.type === selectedType;
    return matchesSearch && matchesType;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <div className="text-red-600">{error}</div>
        {error === 'Please sign in to access exercises' && (
          <a
            href="/auth/login"
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Sign In
          </a>
        )}
      </div>
    );
  }

  return (
    <>
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <Dumbbell className="text-indigo-600" size={32} />
          <h1 className="text-3xl font-bold text-gray-900">Practice Exercises</h1>
        </div>
        <p className="text-gray-600">
          Reinforce your learning with hands-on exercises and interactive visualizations.
        </p>
      </header>

      {activeExercise ? (
        <div className="mb-8">
          <button
            onClick={() => setActiveExercise(null)}
            className="mb-4 text-indigo-600 hover:text-indigo-700 font-medium flex items-center gap-2"
          >
            ← Back to exercises
          </button>
          
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              {activeExercise.title}
            </h2>
            <p className="text-gray-600 mb-6">{activeExercise.description}</p>
            
            {activeExercise.type === 'interactive' && (
              <InteractiveVisualization />
            )}
            
            {activeExercise.type === 'visualization' && (
              <div className="bg-gray-50 rounded-lg p-8 text-center">
                <p className="text-gray-500">Visualization component will be implemented here</p>
              </div>
            )}
            
            {activeExercise.type === 'quiz' && (
              <div className="bg-gray-50 rounded-lg p-8 text-center">
                <p className="text-gray-500">Quiz component will be implemented here</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        <>
          <div className="mb-8 grid gap-4 md:grid-cols-[1fr,auto]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Search exercises..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
              />
            </div>

            <select
              value={selectedType || ''}
              onChange={(e) => setSelectedType(e.target.value || null)}
              className="rounded-lg border border-gray-200 px-4 py-2 focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            >
              <option value="">All Types</option>
              <option value="interactive">Interactive</option>
              <option value="visualization">Visualization</option>
              <option value="quiz">Quiz</option>
            </select>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {filteredExercises.map(exercise => (
              <div
                key={exercise.id}
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-2 rounded-lg ${
                    exercise.type === 'interactive' ? 'bg-purple-100' :
                    exercise.type === 'visualization' ? 'bg-blue-100' :
                    'bg-green-100'
                  }`}>
                    {exercise.type === 'interactive' ? (
                      <Brain className="text-purple-600" size={24} />
                    ) : exercise.type === 'visualization' ? (
                      <BarChart2 className="text-blue-600" size={24} />
                    ) : (
                      <Brain className="text-green-600" size={24} />
                    )}
                  </div>
                </div>

                <h3 className="text-lg font-semibold text-gray-900 mb-2">{exercise.title}</h3>
                <p className="text-gray-600 mb-4">{exercise.description}</p>

                <div className="flex items-center justify-between">
                  <span className={`px-3 py-1 rounded-full text-sm ${
                    exercise.type === 'interactive' ? 'bg-purple-50 text-purple-700' :
                    exercise.type === 'visualization' ? 'bg-blue-50 text-blue-700' :
                    'bg-green-50 text-green-700'
                  }`}>
                    {exercise.type.charAt(0).toUpperCase() + exercise.type.slice(1)}
                  </span>
                  
                  <button
                    onClick={() => setActiveExercise(exercise)}
                    className="flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-medium"
                  >
                    Start Exercise
                    <ArrowRight size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>

          {filteredExercises.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500">No exercises found matching your criteria.</p>
            </div>
          )}
        </>
      )}
    </>
  );
}