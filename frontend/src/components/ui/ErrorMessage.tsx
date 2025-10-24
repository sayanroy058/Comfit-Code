import React from "react";
import { RefreshCw, AlertCircle, X } from "lucide-react";

interface ErrorMessageProps {
  error: string;
  onRetry: () => void;
  onDismiss?: () => void;
  showRetry?: boolean;
  className?: string;
  previousInput?: string;
}

export default function ErrorMessage({
  error,
  onRetry,
  onDismiss,
  showRetry = true,
  className = "",
  previousInput,
}: ErrorMessageProps) {
  return (
    <div
      className={`bg-red-900/20 border border-red-500/30 rounded-lg p-4 ${className}`}
    >
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="text-red-200 font-medium mb-1">Error occurred</div>
          {previousInput && (
            <div className="text-red-300 text-sm mb-2">
              <span className="opacity-70">Previous input:</span> "
              {previousInput}"
            </div>
          )}
          <div className="text-red-300 text-sm mb-3 break-words overflow-hidden">
            {error}
          </div>
          <div className="flex items-center gap-2">
            {showRetry && (
              <button
                onClick={onRetry}
                className="inline-flex items-center gap-2 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded-md transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Retry
              </button>
            )}
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="inline-flex items-center gap-2 px-3 py-1.5 bg-transparent hover:bg-red-800/30 text-red-300 text-sm rounded-md transition-colors border border-red-500/30"
              >
                <X className="w-4 h-4" />
                Dismiss
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
