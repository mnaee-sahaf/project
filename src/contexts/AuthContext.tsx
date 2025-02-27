import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, AuthError } from '@supabase/supabase-js';
import { supabase, checkSupabaseConnection } from '../lib/supabase';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signUp: (email: string, password: string, name: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  error: string | null;
  userName: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [userName, setUserName] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionChecked, setConnectionChecked] = useState(false);

  useEffect(() => {
    const initializeAuth = async () => {
      try {
        // Check Supabase connection first
        const isConnected = await checkSupabaseConnection();
        if (!isConnected) {
          throw new Error('Unable to connect to the server. Please check your internet connection.');
        }
        setConnectionChecked(true);

        const { data: { session } } = await supabase.auth.getSession();
        setUser(session?.user ?? null);
        if (session?.user) {
          await fetchUserName(session.user.id);
        }
      } catch (err) {
        console.error('Auth initialization error:', err);
        setError(err instanceof Error ? err.message : 'Failed to initialize authentication');
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();

    const { data: { subscription } } = supabase.auth.onAuthStateChange(async (_event, session) => {
      setUser(session?.user ?? null);
      if (session?.user) {
        await fetchUserName(session.user.id);
      } else {
        setUserName(null);
      }
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  const fetchUserName = async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from('user_profiles')
        .select('name')
        .eq('user_id', userId)
        .single();
      
      if (error) throw error;
      if (data) {
        setUserName(data.name);
      }
    } catch (err) {
      console.error('Error fetching user name:', err);
      // Don't set error state here as it's not critical
    }
  };

  const handleAuthError = (err: unknown): string => {
    if (err instanceof AuthError) {
      switch (err.status) {
        case 400:
          return 'Invalid email or password';
        case 422:
          return 'Invalid email format';
        case 429:
          return 'Too many attempts. Please try again later';
        case 0:
          return 'Network error. Please check your internet connection';
        default:
          return err.message;
      }
    }
    return 'An unexpected error occurred';
  };

  const signUp = async (email: string, password: string, name: string) => {
    try {
      setError(null);
      if (!connectionChecked) {
        throw new Error('Still checking connection. Please try again.');
      }

      const { data: authData, error: authError } = await supabase.auth.signUp({
        email,
        password,
      });
      if (authError) throw authError;

      if (authData.user) {
        const { error: profileError } = await supabase
          .from('user_profiles')
          .insert([{ user_id: authData.user.id, name }]);
        
        if (profileError) throw profileError;
        setUserName(name);
      }
    } catch (err) {
      const errorMessage = handleAuthError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      setError(null);
      if (!connectionChecked) {
        throw new Error('Still checking connection. Please try again.');
      }

      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (error) throw error;
    } catch (err) {
      const errorMessage = handleAuthError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  };

  const signOut = async () => {
    try {
      setError(null);
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
    } catch (err) {
      const errorMessage = handleAuthError(err);
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  };

  return (
    <AuthContext.Provider value={{ user, userName, loading, signUp, signIn, signOut, error }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}