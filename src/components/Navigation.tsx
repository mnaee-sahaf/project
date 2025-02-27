import React from 'react';
import { Menu, Home, BookOpen, Activity, Award, LogOut, LogIn } from 'lucide-react';
import { NavLink, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export function Navigation() {
  const { user, userName, signOut } = useAuth();

  return (
    <nav className="fixed bottom-0 left-0 w-full bg-white border-t md:relative md:border-t-0 md:border-r md:w-64 md:h-screen">
      <div className="flex justify-around items-center p-4 md:flex-col md:items-start md:space-y-6">
        <div className="hidden md:block mb-8">
          <Link to="/" className="text-2xl font-bold text-indigo-600 hover:text-indigo-700 transition-colors">
            ML Academy
          </Link>
          {userName && (
            <p className="text-sm text-gray-600 mt-2">Welcome, {userName}</p>
          )}
        </div>
        
        <NavItem to="/dashboard" icon={<Home size={20} />} label="Dashboard" />
        <NavItem to="/lessons" icon={<BookOpen size={20} />} label="Lessons" />
        <NavItem to="/practice" icon={<Activity size={20} />} label="Practice" />
        <NavItem to="/progress" icon={<Award size={20} />} label="Progress" />
        
        {user ? (
          <button
            onClick={() => signOut()}
            className="flex items-center space-x-3 p-2 rounded-lg w-full transition-colors text-red-600 hover:bg-red-50"
          >
            <LogOut size={20} />
            <span className="hidden md:inline">Sign Out</span>
          </button>
        ) : (
          <Link
            to="/auth/login"
            className="flex items-center space-x-3 p-2 rounded-lg w-full transition-colors text-indigo-600 hover:bg-indigo-50"
          >
            <LogIn size={20} />
            <span className="hidden md:inline">Sign In</span>
          </Link>
        )}
      </div>
    </nav>
  );
}

function NavItem({ icon, label, to }: { icon: React.ReactNode; label: string; to: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `
        flex items-center space-x-3 p-2 rounded-lg w-full transition-colors
        ${isActive 
          ? 'text-indigo-600 bg-indigo-50 hover:bg-indigo-100' 
          : 'text-gray-600 hover:bg-gray-100'}
      `}
    >
      {icon}
      <span className="hidden md:inline">{label}</span>
    </NavLink>
  );
}