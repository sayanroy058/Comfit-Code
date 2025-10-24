"use server";

import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabaseServer";

export async function requestPasswordReset(formData: FormData) {
  const supabase = await createClient();
  const email = formData.get("email") as string;
  const captchaToken = formData.get("captchaToken") as string;

  // redirect to same URL as in redirect url in authentication configuration
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `http://localhost:3000/auth/reset-password`,
    captchaToken: captchaToken,
  });

  if (error) {
    redirect(
      `/auth/forgot-password?error=${encodeURIComponent(error.message)}`
    );
  }

  redirect(`/auth/forgot-password?sent=true`);
}
