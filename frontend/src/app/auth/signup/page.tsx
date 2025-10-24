"use client";

import { useEffect, useState, useRef, Suspense } from "react";
import Script from "next/script";
import { createClient } from "@/lib/supabaseClient";
import { signup } from "../login/actions";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  AlertCircle,
  ArrowLeft,
  Mail,
  Lock,
  User,
  Phone,
  Eye,
  EyeOff,
} from "lucide-react";
import { FaMicrosoft } from "react-icons/fa";
import HCaptcha from "@hcaptcha/react-hcaptcha";
import { useSearchParams } from "next/navigation";

declare global {
  interface Window {
    handleSignInWithGoogle: (response: { credential: string }) => void;
    google?: any;
  }
}

function SignupPageContent() {
  const params = useSearchParams();
  const err = params.get("error");

  // for captcha
  const [captchaToken, setCaptchaToken] = useState<string>("");
  const captchaRef = useRef<HCaptcha>(null);

  // to see password
  const [showPassword, setShowPassword] = useState(false);
  const [googleLoaded, setGoogleLoaded] = useState(false);

  const supabase = createClient();
  useEffect(() => {
    window.handleSignInWithGoogle = async (response: any) => {
      const { data, error } = await supabase.auth.signInWithIdToken({
        provider: "google",
        token: response.credential,
      });
      if (error) {
        console.error("Google sign-in error:", error.message);
      } else {
        window.location.assign("/chat");
      }
    };
  }, [supabase]);

  // initialize google button
  useEffect(() => {
    if (window.google && window.google.accounts) {
      setGoogleLoaded(true);
      // force re render
      window.google.accounts.id.renderButton(
        document.getElementById("google-signin-button"),
        {
          type: "standard",
          shape: "rectangular",
          theme: "outline",
          text: "signin_with",
          size: "large",
          logo_alignment: "left",
        }
      );
    }
  }, [googleLoaded]);

  const signInWithAzure = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "azure",
      options: { scopes: "email" },
    });
    if (error) {
      console.error("Azure sign-in error:", error.message);
    }
  };

  const handleGoogleScriptLoad = () => {
    if (window.google && window.google.accounts) {
      setGoogleLoaded(true);
    }
  };

  return (
    <>
      {/* load Google Identity Services library */}
      <Script
        src="https://accounts.google.com/gsi/client"
        strategy="afterInteractive"
        onLoad={handleGoogleScriptLoad}
      />

      <div className="flex min-h-screen bg-background text-foreground items-center justify-center p-4">
        <div className="relative w-full max-w-4xl bg-card rounded-xl shadow-lg border border-border overflow-hidden">
          <div className="absolute top-4 right-4">
            <Link href="/">
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-foreground hover:bg-accent cursor-pointer"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Chat
              </Button>
            </Link>
          </div>

          {/* Two-column layout */}
          <div className="grid grid-cols-1 md:grid-cols-2">
            {/* Sign-up form */}
            <div className="p-8">
              <div className="max-w-md mx-auto">
                <div className="mb-8">
                  <h1 className="text-2xl font-bold text-white">
                    Create your account
                  </h1>
                  <p className="text-gray-400 mt-2">
                    Sign up for unlimited messages
                  </p>
                </div>

                <form action={signup} className="space-y-5">
                  <input
                    type="hidden"
                    name="captchaToken"
                    value={captchaToken}
                  />

                  {err && (
                    <div className="p-3 bg-red-900/30 border border-red-800 rounded-md flex items-center gap-2 text-red-300">
                      <AlertCircle className="h-5 w-5 flex-shrink-0" />
                      <p>{err}</p>
                    </div>
                  )}

                  {/* Full Name */}
                  <div className="space-y-2">
                    <label
                      htmlFor="fullName"
                      className="block text-sm font-medium text-gray-300"
                    >
                      Full Name
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <User className="h-5 w-5 text-gray-500" />
                      </div>
                      <input
                        id="fullName"
                        name="fullName"
                        type="text"
                        autoComplete="name"
                        required
                        className="w-full bg-gray-700 border border-gray-600 text-white pl-10 px-4 py-3 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                        placeholder="Enter your full name"
                      />
                    </div>
                  </div>

                  {/* Email */}
                  <div className="space-y-2">
                    <label
                      htmlFor="email"
                      className="block text-sm font-medium text-gray-300"
                    >
                      Email
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Mail className="h-5 w-5 text-gray-500" />
                      </div>
                      <input
                        id="email"
                        name="email"
                        type="email"
                        required
                        className="w-full bg-gray-700 border border-gray-600 text-white pl-10 px-4 py-3 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                        placeholder="Enter your email"
                      />
                    </div>
                  </div>

                  {/* Password */}
                  <div className="space-y-2">
                    <label
                      htmlFor="password"
                      className="block text-sm font-medium text-gray-300"
                    >
                      Password
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="h-5 w-5 text-gray-500" />
                      </div>
                      <input
                        id="password"
                        name="password"
                        type={showPassword ? "text" : "password"}
                        required
                        className="w-full bg-gray-700 border border-gray-600 text-white pl-10 px-4 py-3 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                        placeholder="Create a strong password"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition-colors"
                      >
                        {showPassword ? (
                          <EyeOff className="h-5 w-5" />
                        ) : (
                          <Eye className="h-5 w-5" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* hCaptcha widget */}
                  <HCaptcha
                    ref={captchaRef}
                    sitekey="61e2f4f0-53a6-444a-b996-e25e08d2f892"
                    onVerify={(token) => setCaptchaToken(token)}
                  />

                  <button
                    type="submit"
                    //disabled={!captchaToken}
                    formAction={signup}
                    className="w-full py-3 px-4 bg-primary hover:bg-primary/90 focus:bg-primary/90 text-primary-foreground font-medium rounded-md transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-gray-800 cursor-pointer"
                  >
                    Create account
                  </button>
                </form>

                {/* Footer link */}
                <div className="mt-6 text-center">
                  <p className="text-sm text-gray-400">
                    Already have an account?{" "}
                    <Link
                      href="/auth/login"
                      className="text-primary hover:text-primary/80 font-medium transition-colors"
                    >
                      Sign in
                    </Link>
                  </p>
                </div>
              </div>
            </div>

            {/* Column 2: Third-party authentication */}
            <div className="bg-muted border-l border-border p-8 flex flex-col">
              <div className="max-w-md mx-auto w-full flex flex-col flex-1 justify-center">
                <h2 className="text-xl font-bold mb-6 text-center">
                  Continue with
                </h2>

                <div className="space-y-4">
                  {/* Google sign-up */}
                  <div className="flex justify-center mb-4">
                    <div
                      id="g_id_onload"
                      data-client_id="847601789956-p3qpu7kmfc9561q4hukcisg1vgk8rj6u.apps.googleusercontent.com"
                      data-context="signup"
                      data-ux_mode="popup"
                      data-callback="handleSignInWithGoogle"
                      data-nonce=""
                      data-auto_select="true"
                      data-itp_support="true"
                      data-use_fedcm_for_prompt="false"
                    ></div>

                    {googleLoaded ? (
                      <div id="google-signin-button"></div>
                    ) : (
                      <div className="w-full h-12 bg-input rounded animate-pulse flex items-center justify-center">
                        <span className="text-muted-foreground">
                          Loading Google Sign-in...
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Microsoft sign-in 
                  - TODO: need to fix the css to match googles*/}
                  <button
                    onClick={signInWithAzure}
                    className="inline-flex items-center justify-center w-5/10
                    ml-24 bg-white border border-gray-300 rounded shadow-sm
                    text-sm font-medium text-gray-700
                    hover:bg-gray-50 hover:border-gray-400
                    focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary
                    transition-all duration-200 ease-in-out
                    px-1 py-2.5 cursor-pointer"
                  >
                    <FaMicrosoft className="w-5 h-5 text-primary" />
                    <span className="ml-2">Sign in with Microsoft</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default function SignupPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-screen text-gray-400">
          Loading...
        </div>
      }
    >
      <SignupPageContent />
    </Suspense>
  );
}
