"use client";

import React, { useState, useEffect, useRef } from "react";
import { createClient } from "@/lib/supabaseClient";
import { Button } from "@/components/ui/button";
import HCaptcha from "@hcaptcha/react-hcaptcha";

interface ManageProfileProps {
  isOpen: boolean;
  onClose: () => void;
  user: {
    id?: string;
    email?: string | null;
    user_metadata?: {
      full_name?: string;
      lastNameUpdate?: string;
      lastCredsUpdate?: string;
    };
  };
}

export default function ManageProfile({
  isOpen,
  onClose,
  user,
}: ManageProfileProps) {
  if (!isOpen) return null;
  // form fields
  const [fullName, setFullName] = useState(user.user_metadata?.full_name || "");
  const [email, setEmail] = useState(user.email ?? "");
  const [newPassword, setNewPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState(""); // reauth

  // feedback
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const supabase = createClient();

  // for captcha
  const [captchaToken, setCaptchaToken] = useState<string>("");
  const captchaRef = useRef<HCaptcha>(null);

  // countdown - 15 minutes for changing name, 24 hours for changing email/password
  const MIN_NAME_COOLDOWN = 15 * 60 * 1000; // 15 minutes
  const MIN_CREDS_COOLDOWN = 24 * 60 * 60 * 1000; // 24 hours
  const now = Date.now();

  const lastNameTs = user.user_metadata?.lastNameUpdate
    ? new Date(user.user_metadata.lastNameUpdate).getTime()
    : 0;
  const lastCredsTs = user.user_metadata?.lastCredsUpdate
    ? new Date(user.user_metadata.lastCredsUpdate).getTime()
    : 0;

  const nameCooldownLeft = Math.max(0, MIN_NAME_COOLDOWN - (now - lastNameTs));
  const credsCooldownLeft = Math.max(
    0,
    MIN_CREDS_COOLDOWN - (now - lastCredsTs)
  );

  useEffect(() => {
    if (isOpen) {
      setFullName(user.user_metadata?.full_name || "");
      setEmail(user.email || "");
      setNewPassword("");
      setError(null);
    }
  }, [isOpen, user]);

  const handleSubmit = async (token: string) => {
    setLoading(true);
    setError(null);

    // check name cooldown
    if (fullName !== user.user_metadata?.full_name && nameCooldownLeft > 0) {
      setError(
        `Name cooldown: try again in ${Math.ceil(
          nameCooldownLeft / 60000
        )} minutes`
      );
      return;
    }

    // check creds cooldown
    if ((email !== user.email || newPassword) && credsCooldownLeft > 0) {
      setError(
        `Creds cooldown: try again in ${Math.ceil(
          credsCooldownLeft / 3600000
        )} hours`
      );
      return;
    }

    // if new password require curr password first
    if (newPassword && !currentPassword) {
      setError("Please enter your current password to update password");
      return;
    }

    setLoading(true);

    try {
      if (newPassword) {
        const { error: signinErr } = await supabase.auth.signInWithPassword({
          email,
          password: currentPassword,
          options: { captchaToken: token },
        });
        if (signinErr) throw signinErr;
      }

      // Update auth user data
      const updatePayload: any = {
        data: {
          full_name: fullName,
          ...(fullName !== user.user_metadata?.full_name && {
            last_name_update: new Date().toISOString(),
          }),
          ...((email !== user.email || newPassword) && {
            lastCredsUpdate: new Date().toISOString(),
          }),
        },
        ...(email !== user.email && { email }),
        ...(newPassword && { password: newPassword }),
      };

      const { error: updateErr } = await supabase.auth.updateUser(
        updatePayload
      );
      if (updateErr) throw updateErr;

      // Also update the profiles table
      const { error: profileError } = await supabase
        .from("profiles")
        .update({
          full_name: fullName,
          updated_at: new Date().toISOString(),
        })
        .eq("id", user.id);

      if (profileError) {
        console.error("Profile update error:", profileError);
        // Don't throw here as auth update was successful
      }

      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to update profile");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-800 rounded-lg shadow-lg w-full max-w-md mx-4 p-6">
        <h1 className="text-lg font-semibold text-foreground pb-4">
          Modify Profile
        </h1>
        {error && (
          <p className="mb-4 text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
        <form className="space-y-4">
          <div>
            <label
              htmlFor="fullName"
              className="block text-sm font-medium text-muted-foreground mb-1"
            >
              Display Name
            </label>
            <input
              id="fullName"
              type="text"
              autoFocus
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              disabled={nameCooldownLeft > 0 || loading}
              className="w-full bg-input border border-border text-foreground px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-primary"
            />
            {nameCooldownLeft > 0 && (
              <p className="text-xs text-red-600">
                You can update your name again in{" "}
                {Math.ceil(nameCooldownLeft / 60000)} minutes.
              </p>
            )}
          </div>

          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-muted-foreground mb-1"
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={credsCooldownLeft > 0 || loading}
              className="w-full bg-input border border-border text-muted-foreground px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-muted-foreground mb-1"
            >
              New Password{" "}
              <span className="text-xs text-muted-foreground">
                (leave blank to keep current)
              </span>
            </label>
            <input
              id="password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              disabled={credsCooldownLeft > 0 || loading}
              className="w-full bg-input border border-border text-foreground px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-primary"
            />
            {credsCooldownLeft > 0 && (
              <p className="text-xs text-red-600">
                You can update your password/email again in{" "}
                {Math.ceil(credsCooldownLeft / 3600000)} hours.
              </p>
            )}
          </div>

          {newPassword && (
            <>
              <label className="block text-sm font-medium text-muted-foreground mb-1">
                Current Password{" "}
                <span className="text-xs text-muted-foreground">
                  (Enter current password to update password)
                </span>
              </label>
              <input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                disabled={loading}
                className="w-full bg-input border border-border text-foreground px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </>
          )}

          {/* hCaptcha widget */}
          <HCaptcha
            ref={captchaRef}
            sitekey="61e2f4f0-53a6-444a-b996-e25e08d2f892"
            size="invisible"
            onVerify={(token) => handleSubmit(token)}
          />

          <div className="flex justify-end space-x-2">
            <Button
              type="button"
              variant="secondary"
              onClick={onClose}
              disabled={loading}
              className="cursor-pointer"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              onClick={(e) => {
                e.preventDefault();
                if (captchaToken) handleSubmit(captchaToken);
                else captchaRef.current?.execute();
              }}
              disabled={loading}
              className="cursor-pointer"
            >
              {loading ? "Saving…" : "Save"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
