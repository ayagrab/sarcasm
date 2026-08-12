import type { MethodPrediction } from "@/lib/api";
import StatusBadge from "@/components/StatusBadge";

export default function MethodCard({
  prediction,
  highlight,
}: {
  prediction: MethodPrediction;
  highlight: "agree" | "disagree" | "neutral";
}) {
  const isSarcastic = prediction.label === "sarcastic";
  const borderClass =
    highlight === "disagree"
      ? "border-red-300 dark:border-red-800"
      : highlight === "agree"
        ? "border-green-300 dark:border-green-800"
        : "border-black/10 dark:border-white/10";

  return (
    <div className={`rounded-lg border ${borderClass} p-4`}>
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-medium">{prediction.display_name}</h3>
        <StatusBadge status={prediction.status} />
      </div>

      {prediction.status === "AVAILABLE" && prediction.label && (
        <div className="mt-3">
          <p
            className={`text-lg font-semibold ${
              isSarcastic
                ? "text-orange-600 dark:text-orange-400"
                : "text-blue-600 dark:text-blue-400"
            }`}
          >
            {isSarcastic ? "Sarcastic" : "Not Sarcastic"}
          </p>
          <div className="mt-1 flex gap-4 text-sm text-black/60 dark:text-white/60">
            {prediction.confidence !== null && (
              <span>Confidence: {(prediction.confidence * 100).toFixed(0)}%</span>
            )}
            {prediction.runtime_seconds !== null && (
              <span>Runtime: {prediction.runtime_seconds.toFixed(2)}s</span>
            )}
          </div>
        </div>
      )}

      {prediction.status !== "AVAILABLE" && (
        <p className="mt-3 text-sm text-black/50 dark:text-white/50">
          {prediction.error
            ? `Error: ${prediction.error}`
            : "This method isn't ready to serve predictions yet."}
        </p>
      )}
    </div>
  );
}
