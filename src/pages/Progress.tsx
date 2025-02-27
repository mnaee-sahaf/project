import React from 'react';
import { Award, Target, Clock, Brain, BarChart3, Trophy, Zap, BookOpen } from 'lucide-react';
import { useProgress } from '../hooks/useProgress';

function StatCard({ icon: Icon, label, value, className = '' }: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  className?: string;
}) {
  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-6 ${className}`}>
      <div className="flex items-center gap-4">
        <div className="p-3 bg-indigo-50 rounded-lg">
          <Icon className="text-indigo-600" size={24} />
        </div>
        <div>
          <p className="text-sm text-gray-600">{label}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ value, max }: { value: number; max: number }) {
  const percentage = (value / max) * 100;
  return (
    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
      <div 
        className="h-full bg-indigo-600 rounded-full"
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}

export function Progress() {
  const { progress, recentLessons, loading, error } = useProgress();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-red-600">Error: {error}</div>
      </div>
    );
  }

  if (!progress) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-gray-600">No progress data available</div>
      </div>
    );
  }

  const {
    completed_lessons = [],
    hours_spent = 0,
    current_streak = 0,
    average_score = 0
  } = progress;

  return (
    <>
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <BarChart3 className="text-indigo-600" size={32} />
          <h1 className="text-3xl font-bold text-gray-900">Your Progress</h1>
        </div>
        <p className="text-gray-600">Track your learning journey and achievements.</p>
      </header>

      <div className="grid gap-6 mb-8 md:grid-cols-2 lg:grid-cols-4">
        <StatCard 
          icon={BookOpen}
          label="Completed Lessons"
          value={completed_lessons.length}
        />
        <StatCard 
          icon={Clock}
          label="Hours Spent Learning"
          value={hours_spent.toFixed(1)}
        />
        <StatCard 
          icon={Zap}
          label="Current Streak"
          value={`${current_streak} days`}
        />
        <StatCard 
          icon={Brain}
          label="Average Score"
          value={`${average_score.toFixed(1)}%`}
        />
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        <section>
          <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
          {recentLessons && recentLessons.length > 0 ? (
            <div className="space-y-4">
              {recentLessons.map(lesson => (
                <div 
                  key={lesson.id}
                  className="bg-white rounded-lg shadow-sm border border-gray-200 p-4"
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-medium text-gray-900">{lesson.title}</h3>
                    <span className="text-sm text-gray-500">
                      {new Date(lesson.completed_at).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className={`px-2 py-1 rounded-full text-sm
                      ${lesson.difficulty === 'beginner' ? 'bg-green-100 text-green-800' :
                        lesson.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'}`}>
                      {lesson.difficulty}
                    </span>
                    <div className="flex items-center gap-2">
                      <Trophy 
                        size={16} 
                        className={lesson.score >= 90 ? 'text-yellow-500' : 'text-gray-400'} 
                      />
                      <span className="font-medium">{lesson.score}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 bg-white rounded-lg border border-gray-200">
              <p className="text-gray-500">No lessons completed yet</p>
            </div>
          )}
        </section>

        <section>
          <h2 className="text-xl font-semibold mb-4">Achievements</h2>
          <div className="space-y-4">
            {[
              {
                id: 1,
                title: 'Fast Learner',
                description: 'Complete 3 lessons in one day',
                icon: Zap,
                earned: (recentLessons || []).length >= 3,
                progress: Math.min(((recentLessons || []).length / 3) * 100, 100),
              },
              {
                id: 2,
                title: 'Perfect Score',
                description: 'Get 100% on any lesson quiz',
                earned: (recentLessons || []).some(lesson => lesson.score === 100),
                progress: Math.max(...(recentLessons || []).map(lesson => lesson.score), 0),
                icon: Trophy,
              },
              {
                id: 3,
                title: 'Consistent Learner',
                description: 'Maintain a 7-day learning streak',
                icon: Target,
                earned: current_streak >= 7,
                progress: (current_streak / 7) * 100,
              },
            ].map(achievement => (
              <div 
                key={achievement.id}
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-4"
              >
                <div className="flex items-start gap-4">
                  <div className={`p-2 rounded-lg ${
                    achievement.earned ? 'bg-green-100' : 'bg-gray-100'
                  }`}>
                    <achievement.icon 
                      size={24}
                      className={achievement.earned ? 'text-green-600' : 'text-gray-400'}
                    />
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="font-medium text-gray-900">{achievement.title}</h3>
                        <p className="text-sm text-gray-600">{achievement.description}</p>
                      </div>
                      {achievement.earned && (
                        <Award className="text-yellow-500" size={20} />
                      )}
                    </div>
                    <ProgressBar value={achievement.progress} max={100} />
                    <p className="text-sm text-gray-500 mt-2">
                      {achievement.progress.toFixed(0)}% Complete
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}