"use client";

import { useState } from "react";
import { ApiError, compare, type CompareResponse } from "@/lib/api";
import ExampleSentences from "@/components/ExampleSentences";
import MethodCard from "@/components/MethodCard";

export default function ResearchModePage() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleCompare() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await compare(text);
      setResult(response);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while comparing approaches.",
      );
    } finally {
      setLoading(false);
    }
  }

  const availableLabels = new Set(
    (result?.predictions ?? [])
      .filter((p) => p.status === "AVAILABLE" && p.label)
      .map((p) => p.label),
  );
  const hasDisagreement = availableLabels.size > 1;

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Research Mode
        </h1>
        <p className="mt-2 text-black/60 dark:text-white/60">
          Compare predictions from every approach implemented in this
          project, side by side, on the same sentence.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. Oh great, my flight was cancelled again."
          rows={3}
          maxLength={2000}
          className="w-full resize-none rounded-lg border border-black/15 bg-transparent p-3 text-base outline-none focus:border-black/40 dark:border-white/20 dark:focus:border-white/50"
        />
        <ExampleSentences onPick={setText} />
        <button
          type="button"
          onClick={handleCompare}
          disabled={loading || !text.trim()}
          className="self-start rounded-lg bg-black px-5 py-2.5 font-medium text-white transition disabled:opacity-40 dark:bg-white dark:text-black"
        >
          {loading ? "Comparing..." : "Compare all methods"}
        </button>
      </div>

      {error && (
        <p className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </p>
      )}

      {result && (
        <div className="flex flex-col gap-4">
          {result.predictions.filter((p) => p.status === "AVAILABLE").length >=
            2 && (
            <p
              className={`rounded-lg border p-3 text-sm ${
                hasDisagreement
                  ? "border-red-300 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300"
                  : "border-green-300 bg-green-50 text-green-800 dark:border-green-900 dark:bg-green-950/40 dark:text-green-300"
              }`}
            >
              {hasDisagreement
                ? "The available methods disagree on this sentence."
                : "All available methods agree on this sentence."}
            </p>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            {result.predictions.map((prediction) => (
              <MethodCard
                key={prediction.method}
                prediction={prediction}
                highlight={
                  prediction.status !== "AVAILABLE" || !hasDisagreement
                    ? "neutral"
                    : "disagree"
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
