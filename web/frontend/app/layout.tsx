import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Nav from "@/components/Nav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Sarcasm Detector",
  description:
    "Classify whether an English sentence is sarcastic, and compare predictions across every approach built in this research project.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Nav />
        <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-10">
          {children}
        </main>
        <footer className="mx-auto w-full max-w-3xl px-4 py-6 text-xs text-black/40 dark:text-white/40">
          Sarcasm Detector -- a research project demo. Predictions use the
          exact frozen inference configurations selected during
          experimentation; this page never re-tunes a model against
          sentences typed here.
        </footer>
      </body>
    </html>
  );
}
