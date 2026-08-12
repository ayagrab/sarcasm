"use client";

import { useEffect, useState } from "react";
import { listMethods, type MethodInfo } from "@/lib/api";
import StatusBadge from "@/components/StatusBadge";

const APPROACH_ORDER = [
  "tfidf",
  "qwen_zero_shot",
  "qwen_few_shot",
  "qwen_reasoning",
  "dspy",
  "deberta",
];

export default function AboutPage() {
  const [methods, setMethods] = useState<MethodInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listMethods()
      .then((data) => {
        const ordered = [...data].sort(
          (a, b) => APPROACH_ORDER.indexOf(a.method) - APPROACH_ORDER.indexOf(b.method),
        );
        setMethods(ordered);
      })
      .catch(() => setError("Couldn't reach the backend to load method info."));
  }, []);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          About the approaches
        </h1>
        <p className="mt-2 text-black/60 dark:text-white/60">
          This project compares six fundamentally different approaches to
          sarcasm detection, evaluated under the same conditions on the same
          data. See the project&apos;s <code>PROJECT_SUMMARY.md</code> for
          the full methodology.
        </p>
      </div>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      <div className="flex flex-col gap-4">
        {(methods ?? []).map((method) => (
          <div
            key={method.method}
            className="rounded-lg border border-black/10 p-4 dark:border-white/10"
          >
            <div className="flex items-center justify-between gap-2">
              <h2 className="font-medium">{method.display_name}</h2>
              <StatusBadge status={method.status} />
            </div>
            <p className="mt-2 text-sm text-black/60 dark:text-white/60">
              {method.description}
            </p>
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-black/10 p-4 dark:border-white/10">
        <h2 className="font-medium">Final TEST results</h2>
        <p className="mt-2 text-sm text-black/60 dark:text-white/60">
          Pending: TEST is deliberately kept sealed until every method above
          has a frozen configuration selected purely from TRAIN/DEV results
          (see the project&apos;s TEST-sealing policy). This page will show
          the real, final TEST metrics for each frozen method here once
          Stage B legitimately completes -- never placeholder or estimated
          numbers.
        </p>
      </div>
    </div>
  );
}
