import Link from "next/link";

export default function Nav() {
  return (
    <header className="border-b border-black/10 dark:border-white/10">
      <nav className="mx-auto flex max-w-3xl items-center justify-between px-4 py-4">
        <Link href="/" className="text-lg font-semibold tracking-tight">
          Sarcasm Detector
        </Link>
        <div className="flex gap-4 text-sm">
          <Link
            href="/"
            className="rounded-md px-3 py-1.5 hover:bg-black/5 dark:hover:bg-white/10"
          >
            Detector
          </Link>
          <Link
            href="/research"
            className="rounded-md px-3 py-1.5 hover:bg-black/5 dark:hover:bg-white/10"
          >
            Research Mode
          </Link>
          <Link
            href="/about"
            className="rounded-md px-3 py-1.5 hover:bg-black/5 dark:hover:bg-white/10"
          >
            About the approaches
          </Link>
        </div>
      </nav>
    </header>
  );
}
