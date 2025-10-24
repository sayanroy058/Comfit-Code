import { createClient } from "@/lib/supabaseClient";

export interface Profile {
  id: string;
  full_name: string;
  created_at: string;
  updated_at: string;
}

export async function getUserProfile(userId: string): Promise<Profile | null> {
  const supabase = createClient();

  const { data, error } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", userId)
    .single();

  if (error) {
    console.error("Error fetching profile:", error);
    return null;
  }

  return data;
}

export async function updateUserProfile(
  userId: string,
  updates: Partial<Profile>
): Promise<boolean> {
  const supabase = createClient();

  const { error } = await supabase
    .from("profiles")
    .update(updates)
    .eq("id", userId);

  if (error) {
    console.error("Error updating profile:", error);
    return false;
  }

  return true;
}

export async function ensureProfileExists(
  userId: string,
  fullName: string
): Promise<boolean> {
  const supabase = createClient();

  const { error } = await supabase.from("profiles").upsert(
    {
      id: userId,
      full_name: fullName,
    },
    {
      onConflict: "id",
    }
  );

  if (error) {
    console.error("Error ensuring profile exists:", error);
    return false;
  }

  return true;
}
