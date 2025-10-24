"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabaseServer";

export async function login(formData: FormData) {
  const supabase = await createClient();

  const email = formData.get("email") as string;
  const password = formData.get("password") as string;
  //const captchaToken = formData.get("captchaToken") as string;

  const { error } = await supabase.auth.signInWithPassword({
    email,
    password,
    // add captcha token back here
  });

  // catch errors
  if (error) {
    let msg: string;
    if (error.code === "invalid_credentials") {
      msg = "Email or password is incorrect.";
    } else if (error.code === "email_not_confirmed") {
      msg = "Email address needs to be confirmed.";
    } else if (error.code === "captcha_failed") {
      msg = "CAPTCHA challenge could not be verified.";
    } else if (error.status && error.status >= 500) {
      msg = "Server error - please try again later.";
    } else {
      msg = error.message;
    }

    redirect(`/auth/login?error=${encodeURIComponent(msg)}`);
  }

  revalidatePath("/", "layout");
  redirect("/");
}

export async function signup(formData: FormData) {
  const supabase = await createClient();

  // might need to check for input sanitization here
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;
  const fullName = formData.get("fullName") as string;
  //const captchaToken = formData.get("captchaToken") as string;

  /*
  const { error } = await supabase.auth.signUp({
    email,
    password,
    options: { data: { fullName, phone }, captchaToken },
  });
  */

  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        full_name: fullName, // Use full_name to match the trigger function
        fullName: fullName, // Keep both for compatibility
      },
    },
  });

  if (error) {
    let msg: string;
    if (error.code === "email_exists") {
      msg = "Email address already exists, sign up instead.";
    } else if (error.code === "captcha_failed") {
      msg = "CAPTCHA challenge could not be verified.";
    } else if (error.status && error.status >= 500) {
      msg = "Server error - please try again later.";
    } else {
      msg = error.message;
    }
    redirect(`/auth/login?error=${encodeURIComponent(msg)}`);
  }

  // If user was created successfully, ensure profile exists
  if (data.user) {
    try {
      // Try to insert profile if it doesn't exist (fallback in case trigger fails)
      const { error: profileError } = await supabase.from("profiles").upsert(
        {
          id: data.user.id,
          full_name: fullName,
        },
        {
          onConflict: "id",
        }
      );

      if (profileError) {
        console.error("Profile creation error:", profileError);
      }
    } catch (profileErr) {
      console.error("Profile creation failed:", profileErr);
    }
  }

  revalidatePath("/", "layout");
  redirect("/auth/confirm");
}
