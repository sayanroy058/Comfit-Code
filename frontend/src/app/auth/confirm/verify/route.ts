import { type EmailOtpType } from "@supabase/supabase-js";
import { type NextRequest } from "next/server";

import { createClient } from "@/lib/supabaseServer";
import { redirect } from "next/navigation";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const token_hash = searchParams.get("token_hash");
  const type = searchParams.get("type") as EmailOtpType | null;
  const next = searchParams.get("next") ?? "/";

  if (token_hash && type) {
    const supabase = await createClient();
    const { data, error } = await supabase.auth.verifyOtp({
      type,
      token_hash,
    });

    if (!error && data.user) {
      // Ensure profile exists after email confirmation
      try {
        const fullName =
          data.user.user_metadata?.full_name ||
          data.user.user_metadata?.fullName ||
          "Anonymous";

        await supabase.from("profiles").upsert(
          {
            id: data.user.id,
            full_name: fullName,
          },
          {
            onConflict: "id",
          }
        );
      } catch (profileError) {
        console.error(
          "Profile creation error during verification:",
          profileError
        );
      }

      // for password reset
      if (type === "recovery") {
        redirect("/auth/reset-password");
      }
      // redirect to main chat
      redirect("/chat");
    }
  }

  redirect("/error");
}
