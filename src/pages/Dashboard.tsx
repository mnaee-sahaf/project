import React from 'react';
import { InteractiveVisualization } from '../components/InteractiveVisualization';
import { LessonCard } from '../components/LessonCard';
import type { Lesson } from '../types';

const sampleLessons: Lesson[] = [
  {
    id: '1',
    title: 'Introduction to Machine Learning',
    description: 'Learn the fundamental concepts of machine learning and its applications.',
    category: 'basics',
    difficulty: 'beginner',
    completed: false,
  },
  {
    id: '2',
    title: 'Linear Regression Fundamentals',
    description: 'Understand how linear regression works and its practical applications.',
    category: 'regression',
    difficulty: 'beginner',
    completed: false,
  },
  {
    id: '3',
    title: 'K-Nearest Neighbors (KNN)',
    description: 'Explore the KNN algorithm and its use in classification problems.',
    category: 'classification',
    difficulty: 'intermediate',
    completed: false,
  },
];

export function Dashboard() {
  return (
    <>
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Welcome back!</h1>
        <p className="text-gray-600 mt-2">Continue your journey into machine learning.</p>
      </header>

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <section>
          <h2 className="text-xl font-semibold mb-4">Your Current Lesson</h2>
          <InteractiveVisualization />
        </section>

        <section>
          <h2 className="text-xl font-semibold mb-4">Next Up</h2>
          <div className="space-y-4">
            {sampleLessons.map(lesson => (
              <LessonCard
                key={lesson.id}
                lesson={lesson}
                onClick={() => console.log('Lesson clicked:', lesson.id)}
              />
            ))}
          </div>
        </section>
      </div>
    </>
  );
}