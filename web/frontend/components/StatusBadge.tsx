import type { ModelStatus } from "@/lib/api";

const STATUS_LABEL: Record<ModelStatus, string> = {
  AVAILABLE: "Available",
  NOT_TRAINED_YET: "Not trained yet",
  NOT_FROZEN_YET: "Not frozen yet",
  UNAVAILABLE: "Unavailable here",
};

const STATUS_CLASS: Record<ModelStatus, string> = {
  AVAILABLE:
    "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  NOT_TRAINED_YET:
    "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  NOT_FROZEN_YET:
    "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  UNAVAILABLE: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
};

export default function StatusBadge({ status }: { status: ModelStatus }) {
  return (
    <span
      className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_CLASS[status]}`}
    >
      {STATUS_LABEL[status]}
    </span>
  );
}
