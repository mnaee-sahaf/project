import React, { useState } from 'react';
import type { Quiz as QuizType, Question } from '../types';
import { CheckCircle, XCircle } from 'lucide-react';

interface QuizProps {
  quiz: QuizType;
  onComplete: (score: number) => void;
}

export function Quiz({ quiz, onComplete }: QuizProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [completed, setCompleted] = useState(false);

  const handleAnswer = (answerIndex: number) => {
    setSelectedAnswer(answerIndex);
    setShowExplanation(true);

    if (answerIndex === quiz.questions[currentQuestion].correctAnswer) {
      setCorrectAnswers(prev => prev + 1);
    }
  };

  const handleNext = () => {
    if (currentQuestion < quiz.questions.length - 1) {
      setCurrentQuestion(prev => prev + 1);
      setSelectedAnswer(null);
      setShowExplanation(false);
    } else {
      setCompleted(true);
      const score = (correctAnswers / quiz.questions.length) * 100;
      onComplete(score);
    }
  };

  const question = quiz.questions[currentQuestion];

  if (completed) {
    const score = (correctAnswers / quiz.questions.length) * 100;
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-4">Quiz Completed!</h2>
        <p className="text-lg mb-4">
          You got {correctAnswers} out of {quiz.questions.length} questions correct.
        </p>
        <div className="text-3xl font-bold mb-8">
          Score: {score.toFixed(1)}%
        </div>
        <div className={`text-lg font-semibold ${
          score >= 70 ? 'text-green-600' : 'text-yellow-600'
        }`}>
          {score >= 70 ? 'Great job! You\'ve passed the quiz!' : 'Keep practicing to improve your score!'}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <span className="text-sm text-gray-500">
            Question {currentQuestion + 1} of {quiz.questions.length}
          </span>
          <span className="text-sm text-gray-500">
            Score: {correctAnswers} / {quiz.questions.length}
          </span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full">
          <div
            className="h-full bg-indigo-600 rounded-full transition-all duration-300"
            style={{ width: `${((currentQuestion + 1) / quiz.questions.length) * 100}%` }}
          />
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
        <h3 className="text-xl font-semibold mb-6">{question.text}</h3>
        
        <div className="space-y-4">
          {question.options.map((option, index) => (
            <button
              key={index}
              onClick={() => !showExplanation && handleAnswer(index)}
              disabled={showExplanation}
              className={`w-full text-left p-4 rounded-lg border transition-all ${
                showExplanation
                  ? index === question.correctAnswer
                    ? 'border-green-500 bg-green-50'
                    : selectedAnswer === index
                    ? 'border-red-500 bg-red-50'
                    : 'border-gray-200'
                  : 'border-gray-200 hover:border-indigo-500 hover:bg-indigo-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <span>{option}</span>
                {showExplanation && index === question.correctAnswer && (
                  <CheckCircle className="text-green-500" size={20} />
                )}
                {showExplanation && selectedAnswer === index && index !== question.correctAnswer && (
                  <XCircle className="text-red-500" size={20} />
                )}
              </div>
            </button>
          ))}
        </div>
      </div>

      {showExplanation && (
        <div className={`p-4 rounded-lg mb-6 ${
          selectedAnswer === question.correctAnswer
            ? 'bg-green-50 border border-green-200'
            : 'bg-red-50 border border-red-200'
        }`}>
          <p className="font-medium mb-2">
            {selectedAnswer === question.correctAnswer ? 'Correct!' : 'Incorrect'}
          </p>
          <p>{question.explanation}</p>
        </div>
      )}

      {showExplanation && (
        <button
          onClick={handleNext}
          className="w-full py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
        >
          {currentQuestion < quiz.questions.length - 1 ? 'Next Question' : 'Complete Quiz'}
        </button>
      )}
    </div>
  );
}