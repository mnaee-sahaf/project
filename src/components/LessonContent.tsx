import React from 'react';
import { ArrowLeft } from 'lucide-react';
import type { Lesson } from '../types';

interface LessonContentProps {
  lesson: Lesson;
  onBack: () => void;
  onStartQuiz: () => void;
}

export function LessonContent({ lesson, onBack, onStartQuiz }: LessonContentProps) {
  return (
    <div className="max-w-4xl mx-auto">
      <button
        onClick={onBack}
        className="mb-6 flex items-center gap-2 text-indigo-600 hover:text-indigo-700 transition-colors"
      >
        <ArrowLeft size={20} />
        Back to lessons
      </button>

      <article className="prose prose-indigo max-w-none">
        <h1>{lesson.title}</h1>
        <div className="flex gap-4 mb-6">
          <span className={`px-3 py-1 rounded-full text-sm
            ${lesson.difficulty === 'beginner' ? 'bg-green-100 text-green-800' :
              lesson.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'}`}>
            {lesson.difficulty}
          </span>
          <span className="px-3 py-1 rounded-full text-sm bg-indigo-100 text-indigo-800">
            {lesson.category}
          </span>
        </div>

        <div dangerouslySetInnerHTML={{ __html: lesson.content || '' }} />

        <div className="mt-12 flex justify-center">
          <button
            onClick={onStartQuiz}
            className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Take Quiz
          </button>
        </div>
      </article>
    </div>
  );
}