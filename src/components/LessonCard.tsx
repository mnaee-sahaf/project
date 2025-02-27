import React from 'react';
import type { Lesson } from '../types';
import { ArrowRight, CheckCircle } from 'lucide-react';

interface LessonCardProps {
  lesson: Lesson;
  onClick: () => void;
}

export function LessonCard({ lesson, onClick }: LessonCardProps) {
  return (
    <div 
      className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow cursor-pointer"
      onClick={onClick}
    >
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{lesson.title}</h3>
        {lesson.completed && (
          <CheckCircle className="text-green-500" size={20} />
        )}
      </div>
      
      <p className="text-gray-600 mb-4">{lesson.description}</p>
      
      <div className="flex justify-between items-center">
        <span className={`px-3 py-1 rounded-full text-sm
          ${lesson.difficulty === 'beginner' ? 'bg-green-100 text-green-800' :
            lesson.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-800' :
            'bg-red-100 text-red-800'}`}>
          {lesson.difficulty}
        </span>
        
        <button className="text-indigo-600 hover:text-indigo-700 flex items-center gap-1">
          Start Learning
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}