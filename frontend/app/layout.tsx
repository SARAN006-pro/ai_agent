import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "AgentPro — Your AI Assistant",
  description: "AgentPro: A production-quality AI chat assistant powered by FastAPI.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} dark`}>
      <body className="main-container bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
