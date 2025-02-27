export interface Lesson {
  id: string;
  title: string;
  description: string;
  category: 'basics' | 'regression' | 'classification' | 'clustering';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  completed: boolean;
  content?: string;
  quiz?: Quiz;
}

export interface Quiz {
  id: string;
  questions: Question[];
}

export interface Question {
  id: string;
  text: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

export interface Exercise {
  id: string;
  lessonId: string;
  title: string;
  description: string;
  type: 'interactive' | 'visualization' | 'quiz';
  completed: boolean;
}

export interface UserProgress {
  completedLessons: string[];
  currentLesson: string | null;
  score: number;
}