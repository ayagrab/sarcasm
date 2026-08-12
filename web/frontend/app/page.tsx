"use client";

import { useState } from "react";
import { ApiError, predict, type PredictResponse } from "@/lib/api";
import ExampleSentences from "@/components/ExampleSentences";

export default function SimpleModePage() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await predict(text);
      setResult(response);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        setError(
          "The production classifier isn't available right now (Stage B may still be in progress). Try again later.",
        );
      } else {
        setError("Something went wrong while classifying that sentence.");
      }
    } finally {
      setLoading(false);
    }
  }

  const isSarcastic = result?.label === "sarcastic";

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Sarcasm Detector
        </h1>
        <p className="mt-2 text-black/60 dark:text-white/60">
          Enter an English sentence and find out whether it reads as
          sarcastic.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. Oh wonderful, another meeting that could have been an email."
          rows={4}
          maxLength={2000}
          className="w-full resize-none rounded-lg border border-black/15 bg-transparent p-3 text-base outline-none focus:border-black/40 dark:border-white/20 dark:focus:border-white/50"
        />
        <ExampleSentences onPick={setText} />
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={loading || !text.trim()}
          className="self-start rounded-lg bg-black px-5 py-2.5 font-medium text-white transition disabled:opacity-40 dark:bg-white dark:text-black"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {error && (
        <p className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </p>
      )}

      {result && (
        <div className="rounded-xl border border-black/10 p-6 dark:border-white/10">
          <p
            className={`text-3xl font-bold ${
              isSarcastic
                ? "text-orange-600 dark:text-orange-400"
                : "text-blue-600 dark:text-blue-400"
            }`}
          >
            {isSarcastic ? "Sarcastic" : "Not Sarcastic"}
          </p>
          {result.confidence !== null && (
            <p className="mt-2 text-black/60 dark:text-white/60">
              Confidence: {(result.confidence * 100).toFixed(0)}%
            </p>
          )}
        </div>
      )}
    </div>
  );
}
