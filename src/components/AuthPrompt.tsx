import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { LogIn, UserPlus, X } from 'lucide-react';

export function AuthPrompt() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) {
    return null;
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg p-4 md:p-6">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900">Want to save your progress?</h3>
          <p className="text-gray-600">Sign in or create an account to track your learning journey.</p>
        </div>
        <div className="flex items-center gap-4">
          <Link
            to="/auth/login"
            className="inline-flex items-center gap-2 px-4 py-2 border border-transparent rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 transition-colors"
          >
            <LogIn size={20} />
            Sign In
          </Link>
          <Link
            to="/auth/signup"
            className="inline-flex items-center gap-2 px-4 py-2 border border-indigo-600 rounded-lg text-indigo-600 hover:bg-indigo-50 transition-colors"
          >
            <UserPlus size={20} />
            Create Account
          </Link>
          <button
            onClick={() => setIsVisible(false)}
            className="p-2 text-gray-400 hover:text-gray-600 transition-colors rounded-full hover:bg-gray-100"
            aria-label="Dismiss"
          >
            <X size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}