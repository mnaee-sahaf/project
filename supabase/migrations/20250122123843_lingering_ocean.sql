/*
  # Add insert policy for user progress

  1. Changes
    - Add INSERT policy for user_progress table if it doesn't exist
    - Safe creation of policies that may already exist
  
  2. Security
    - Maintains existing RLS policies
    - Ensures users can only insert their own progress records
*/

DO $$ 
BEGIN
  -- Create insert policy for user_progress if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 
    FROM pg_policies 
    WHERE schemaname = 'public' 
      AND tablename = 'user_progress' 
      AND policyname = 'Users can insert own progress'
  ) THEN
    CREATE POLICY "Users can insert own progress"
      ON user_progress
      FOR INSERT
      TO authenticated
      WITH CHECK (auth.uid() = user_id);
  END IF;
END $$;