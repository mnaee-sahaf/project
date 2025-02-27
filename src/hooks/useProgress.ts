import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

export interface UserProgress {
  id: string;
  user_id: string;
  completed_lessons: string[];
  hours_spent: number;
  current_streak: number;
  longest_streak: number;
  average_score: number;
  last_activity: string;
  created_at: string;
  updated_at: string;
}

export interface LessonCompletion {
  id: string;
  user_id: string;
  lesson_id: string;
  title: string;
  score: number;
  completed_at: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

export function useProgress() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<UserProgress | null>(null);
  const [recentLessons, setRecentLessons] = useState<LessonCompletion[]>([]);

  useEffect(() => {
    const fetchProgress = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        
        if (!user) {
          throw new Error('User not authenticated');
        }

        // Fetch user progress
        const { data: progressData, error: progressError } = await supabase
          .from('user_progress')
          .select('*')
          .eq('user_id', user.id)
          .maybeSingle();

        if (progressError) throw progressError;

        // If no progress exists, create initial progress record
        if (!progressData) {
          const initialProgress = {
            user_id: user.id,
            completed_lessons: [],
            hours_spent: 0,
            current_streak: 0,
            longest_streak: 0,
            average_score: 0,
            last_activity: new Date().toISOString()
          };

          const { data: newProgress, error: insertError } = await supabase
            .from('user_progress')
            .insert(initialProgress)
            .select()
            .single();

          if (insertError) throw insertError;
          setProgress(newProgress);
        } else {
          setProgress(progressData);
        }

        // Fetch recent lesson completions
        const { data: completions, error: completionsError } = await supabase
          .from('lesson_completions')
          .select('*')
          .eq('user_id', user.id)
          .order('completed_at', { ascending: false })
          .limit(5);

        if (completionsError) throw completionsError;
        setRecentLessons(completions || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchProgress();

    // Subscribe to real-time updates
    const progressSubscription = supabase
      .channel('progress_changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'user_progress',
        },
        (payload) => {
          if (payload.new) {
            setProgress(payload.new as UserProgress);
          }
        }
      )
      .subscribe();

    const completionsSubscription = supabase
      .channel('completion_changes')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'lesson_completions',
        },
        (payload) => {
          setRecentLessons(prev => [payload.new as LessonCompletion, ...prev].slice(0, 5));
        }
      )
      .subscribe();

    return () => {
      progressSubscription.unsubscribe();
      completionsSubscription.unsubscribe();
    };
  }, []);

  return { progress, recentLessons, loading, error };
}