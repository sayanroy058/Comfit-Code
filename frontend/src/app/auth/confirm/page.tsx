import Link from "next/link";
import { Button } from "@/components/ui/button";
import { MailCheck } from "lucide-react";

export default function ConfirmPage() {
  return (
    <div className="flex h-screen bg-background text-foreground items-center justify-center px-4">
      <div className="w-full max-w-sm bg-card rounded-2xl shadow-2xl p-8 space-y-8 text-center">
        {/* Icon */}
        <div className="mx-auto w-20 h-20 flex items-center justify-center bg-accent rounded-full">
          <MailCheck className="w-10 h-10 text-primary" />
        </div>

        {/* Heading */}
        <h1 className="text-3xl font-extrabold">Almost there!</h1>

        {/* Text */}
        <p className="text-muted-foreground leading-relaxed">
          We've sent a verification link to your email. <br />
          Please check your inbox (and spam folder) and click the link to
          complete your sign-up.
        </p>

        {/* Button */}
        <Link href="/auth/login">
          <Button
            variant="secondary"
            className="w-full py-3 text-base font-medium tracking-wide cursor-pointer"
          >
            Back to Log in
          </Button>
        </Link>

        {/* Resend link */}
        <p className="text-sm pt-3 text-muted-foreground">
          Didn't get it? {/*placeholder*/}
          <button className="text-primary hover:underline underline-offset-2 cursor-pointer">
            Resend email
          </button>
        </p>
      </div>
    </div>
  );
}
