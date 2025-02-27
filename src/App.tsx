import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { AuthGuard } from './components/AuthGuard';
import { AuthPrompt } from './components/AuthPrompt';
import { Navigation } from './components/Navigation';
import { Landing } from './pages/Landing';
import { Login } from './pages/auth/Login';
import { SignUp } from './pages/auth/SignUp';
import { Dashboard } from './pages/Dashboard';
import { Lessons } from './pages/Lessons';
import { Practice } from './pages/Practice';
import { Progress } from './pages/Progress';
import { useAuth } from './contexts/AuthContext';

function AppLayout({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  
  return (
    <div className="min-h-screen bg-gray-50 flex">
      <Navigation />
      <main className="flex-1 p-6 md:p-8 pb-24 md:pb-8">
        <div className="max-w-7xl mx-auto">
          {children}
        </div>
      </main>
      {!user && <AuthPrompt />}
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/auth/login" element={<Login />} />
        <Route path="/auth/signup" element={<SignUp />} />
        
        {/* Public routes with auth prompt */}
        <Route path="/dashboard" element={<AppLayout><Dashboard /></AppLayout>} />
        <Route path="/lessons" element={<AppLayout><Lessons /></AppLayout>} />
        
        {/* Protected routes */}
        <Route
          path="/practice"
          element={
            <AuthGuard>
              <AppLayout><Practice /></AppLayout>
            </AuthGuard>
          }
        />
        <Route
          path="/progress"
          element={
            <AuthGuard>
              <AppLayout><Progress /></AppLayout>
            </AuthGuard>
          }
        />
      </Routes>
    </AuthProvider>
  );
}

export default App;