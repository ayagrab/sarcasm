"use client";

export const EXAMPLE_SENTENCES = [
  "Thank you for helping me with the assignment.",
  "Oh fantastic, my flight has been delayed again.",
  "I will meet you at the office tomorrow morning.",
  "Wonderful, another software update that broke everything.",
];

export default function ExampleSentences({
  onPick,
}: {
  onPick: (sentence: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {EXAMPLE_SENTENCES.map((sentence) => (
        <button
          key={sentence}
          type="button"
          onClick={() => onPick(sentence)}
          className="rounded-full border border-black/10 px-3 py-1.5 text-sm text-black/70 transition hover:border-black/30 hover:text-black dark:border-white/15 dark:text-white/70 dark:hover:border-white/40 dark:hover:text-white"
        >
          {sentence}
        </button>
      ))}
    </div>
  );
}
