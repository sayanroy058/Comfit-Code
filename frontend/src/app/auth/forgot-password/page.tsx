"use client";

import { useSearchParams } from "next/navigation";
import { requestPasswordReset } from "./actions";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import HCaptcha from "@hcaptcha/react-hcaptcha";
import React, { useState, useRef, Suspense } from "react";

function ForgotPasswordContent() {
  const params = useSearchParams();
  const error = params.get("error");
  const sent = params.get("sent");

  // for captcha
  const [captchaToken, setCaptchaToken] = useState<string | undefined>();
  const captchaRef = useRef<HCaptcha>(null);

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <div className="w-full max-w-md bg-card rounded-lg shadow-lg p-8 border border-border">
        <h1 className="text-2xl font-bold mb-6 text-foreground text-center">
          Reset your password
        </h1>

        {error && (
          <div className="mb-6 p-3 bg-red-900/30 border border-red-800 rounded-md flex items-center gap-2 text-red-300">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {sent && (
          <div className="mb-6 p-3 bg-green-900/30 border border-green-800 rounded-md flex items-center gap-2 text-green-300">
            <CheckCircle2 className="h-5 w-5 flex-shrink-0" />
            <p>Check your email for a reset link!</p>
          </div>
        )}

        <form action={requestPasswordReset} className="space-y-6">
          <input type="hidden" name="captchaToken" value={captchaToken} />
          <div className="space-y-2">
            <label
              htmlFor="email"
              className="block text-sm font-medium text-muted-foreground"
            >
              Email address
            </label>
            <input
              id="email"
              name="email"
              type="email"
              required
              className="w-full px-4 py-3 bg-input border border-border rounded-md text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
              placeholder="Enter your email"
            />
          </div>

          {/* hCaptcha widget - CHANGE IN DEPLOYMENT*/}
          <HCaptcha
            ref={captchaRef}
            sitekey="61e2f4f0-53a6-444a-b996-e25e08d2f892"
            onVerify={(token) => setCaptchaToken(token)}
          />

          <button
            type="submit"
            className="w-full py-3 px-4 bg-primary hover:bg-primary/90 focus:bg-primary/90 text-primary-foreground font-medium rounded-md transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-gray-800 cursor-pointer"
          >
            Send reset link
          </button>
        </form>

        <div className="mt-6 text-center">
          <a
            href="/auth/login"
            className="text-primary hover:text-primary/80 text-sm transition-colors"
          >
            Return to login
          </a>
        </div>
      </div>
    </div>
  );
}

export default function ForgotPasswordPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      }
    >
      <ForgotPasswordContent />
    </Suspense>
  );
}
