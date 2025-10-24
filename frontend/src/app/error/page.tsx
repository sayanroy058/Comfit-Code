"use client";

import { Button } from "@/components/ui/button";
import { MessageCircle, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

export default function Component() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-slate-900 flex items-center justify-center p-4 overflow-hidden relative">
      {/* animated background */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className={`absolute rounded-full bg-gray-500/10 animate-pulse ${
              mounted ? "opacity-100" : "opacity-0"
            } transition-opacity duration-1000`}
            style={{
              width: `${Math.random() * 300 + 100}px`,
              height: `${Math.random() * 300 + 100}px`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${i * 0.5}s`,
              animationDuration: `${3 + Math.random() * 2}s`,
            }}
          />
        ))}
      </div>

      {/* geometric shapes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(4)].map((_, i) => (
          <div
            key={i}
            className={`absolute border border-gray-400/20 ${
              mounted ? "animate-float" : ""
            }`}
            style={{
              width: "60px",
              height: "60px",
              left: `${20 + i * 20}%`,
              top: `${30 + i * 15}%`,
              animationDelay: `${i * 0.8}s`,
              animationDuration: "6s",
              transform: "rotate(45deg)",
            }}
          />
        ))}
      </div>

      {/* Main content */}
      <div
        className={`text-center z-10 max-w-2xl mx-auto ${
          mounted ? "animate-fade-in-up" : "opacity-0 translate-y-8"
        } transition-all duration-1000 ease-out`}
      >
        {/* Error Number */}
        <div className="relative mb-8">
          <h1 className="text-9xl md:text-[12rem] font-bold text-transparent bg-clip-text bg-gradient-to-r from-gray-400 via-white to-gray-400 leading-none select-none">
            500
          </h1>
          <div className="absolute inset-0 text-9xl md:text-[12rem] font-bold text-gray-500/20 blur-sm leading-none select-none">
            500
          </div>
        </div>

        {/* Error message */}
        <div className="space-y-4 mb-8">
          <h2 className="text-2xl md:text-3xl font-semibold text-white">
            Server Error
          </h2>
          <p className="text-slate-300 text-lg max-w-md mx-auto leading-relaxed">
            Something went wrong, please try again later.
          </p>
        </div>

        {/* Action buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Link href="/chat">
            <Button
              size="lg"
              className="bg-primary hover:bg-primary/90 text-primary-foreground border-0 px-8 py-3 rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-primary/25 group cursor-pointer"
            >
              <MessageCircle className="w-5 h-5 mr-2 group-hover:animate-bounce" />
              Back to Chat
            </Button>
          </Link>

          <Button
            size="lg"
            variant="outline"
            onClick={() => window.location.reload()}
            className="border-slate-600 text-black-300 hover:bg-slate-800 hover:text-white px-8 py-3 rounded-lg transition-all duration-300 hover:scale-105 group cursor-pointer"
          >
            <RefreshCw className="w-5 h-5 mr-2 group-hover:animate-spin" />
            Try Again
          </Button>
        </div>

        {/* help text */}
        <p className="text-slate-500 text-sm mt-8">
          Error Code: 500 • Server Error
        </p>
      </div>

      {/* animations */}
      <style jsx>{`
        @keyframes float {
          0%,
          100% {
            transform: translateY(0px) rotate(45deg);
          }
          50% {
            transform: translateY(-20px) rotate(45deg);
          }
        }

        @keyframes fade-in-up {
          0% {
            opacity: 0;
            transform: translateY(30px);
          }
          100% {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-float {
          animation: float 6s ease-in-out infinite;
        }

        .animate-fade-in-up {
          animation: fade-in-up 1s ease-out forwards;
        }
      `}</style>
    </div>
  );
}
