"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Mic,
  Plus,
  Bot,
  User,
  Zap,
  ChevronDown,
  TrendingUp,
  Newspaper,
  ListTodo,
  RefreshCw,
  AlertTriangle,
  Sparkles,
  Copy,
  Check,
  Search,
  ImageUp,
  Loader2,
  FileUp,
  Camera,
  FileText,
  Sparkle,
  X,
} from "lucide-react";

// ─── UTILS ────────────────────────────────────────────────────────────────────

function getGreeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good Morning";
  if (h < 17) return "Good Afternoon";
  return "Good Evening";
}

function getGreetingEmoji(): string {
  const h = new Date().getHours();
  if (h < 12) return "☀️";
  if (h < 17) return "👋";
  return "🌙";
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatHeaderDate(date: Date): string {
  return date.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function normalizeQuery(q: string): string {
  return q.trim().toLowerCase().replace(/\s+/g, " ");
}

function querySimilarity(a: string, b: string): number {
  if (!a || !b) return 0;
  const aTokens = new Set(a.split(" "));
  const bTokens = new Set(b.split(" "));
  const intersection = [...aTokens].filter((token) => bTokens.has(token)).length;
  const union = new Set([...aTokens, ...bTokens]).size;
  return union === 0 ? 0 : intersection / union;
}

function createMessageId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

const rawApiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "https://ai-agent-595t.onrender.com";
const API_BASE_URL = rawApiBaseUrl
  .replace(/^http:\/\/(.*\.onrender\.com)(\/.*)?$/i, "https://$1$2")
  .replace(/\/$/, "");

function apiUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${normalized}`;
}

type IntentResult = { matched: true; response: string } | { matched: false };

function detectIntent(input: string): IntentResult {
  const t = input.trim().toLowerCase();

  if (/^(hi|hello|hey|howdy|hiya|sup|what'?s up)\b/.test(t)) {
    return {
      matched: true,
      response: `Hello! 👋 I'm AgentPro, your AI assistant. How can I help you today? You can ask me to search the web, analyze data, plan tasks, write content, and much more.`,
    };
  }
  if (/\b(thanks?|thank you|thx|ty|appreciate)\b/.test(t)) {
    return {
      matched: true,
      response: "You're very welcome! 😊 Is there anything else I can assist you with?",
    };
  }
  if (/^help\b/.test(t) || t === "?") {
    return {
      matched: true,
      response: `Here's what I can do for you:\n\n• 🔍 **Search** — Find the latest news, research, or web results\n• 📈 **Analyze** — Stock trends, data patterns, and insights\n• 📝 **Plan** — Tasks, schedules, and project breakdowns\n• ✍️ **Write** — Emails, summaries, code, and documents\n• 💬 **Chat** — Answer questions on any topic\n\nJust type what you need and I'll handle it!`,
    };
  }
  if (/^(bye|goodbye|see you|cya|later)\b/.test(t)) {
    return {
      matched: true,
      response: "Goodbye! 👋 Come back anytime — I'll be here whenever you need me.",
    };
  }
  if (/\b(who are you|what are you|your name)\b/.test(t)) {
    return {
      matched: true,
      response:
        "I'm **AgentPro** — an intelligent AI assistant built to help you search, analyze, plan, and create. Powered by a FastAPI backend, I'm here to handle everything from quick questions to complex tasks.",
    };
  }
  return { matched: false };
}

// ─── SEARCH RESULT PARSING & URL EXTRACTION ────────────────────────────────

interface SearchResult {
  title: string;
  description: string;
  url: string;
  source?: string;
  timestamp?: string;
}

interface StructuredSearchPayload {
  message?: string;
  insights: string[];
  articles: SearchResult[];
}

function extractRealUrl(url: string): string {
  try {
    // Handle DuckDuckGo redirect format: //duckduckgo.com/?uddg=...
    if (url.includes("uddg=")) {
      const parsed = new URL(url.startsWith("//") ? "https:" + url : url);
      const real = parsed.searchParams.get("uddg");
      if (real) return decodeURIComponent(real);
    }
    // Handle regular URLs
    return url.startsWith("http") ? url : "https://" + url;
  } catch {
    return url;
  }
}

function getDomain(url: string): string {
  try {
    const realUrl = extractRealUrl(url);
    const hostname = new URL(realUrl).hostname;
    return hostname.replace("www.", "");
  } catch {
    return "";
  }
}

function parseSearchResults(text: string): SearchResult[] | null {
  // Check if this looks like a search result response
  if (!text.includes("Top results for:")) return null;

  const lines = text.split("\n").filter((line) => line.trim());
  const results: SearchResult[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim() || line.startsWith("Top results for:") || line.startsWith("   ")) {
      continue;
    }

    // Accept numbered, bullet, or plain title lines from backend.
    const titleMatch = line.match(/^\d+\.\s+(.+)$/) ?? line.match(/^[-*]\s+(.+)$/);
    if (titleMatch) {
      const title = titleMatch[1].trim();

      // Look for description and URL in following indented lines
      let description = "";
      let url = "";

      for (let j = i + 1; j < lines.length; j++) {
        const nextLine = lines[j];
        if (!nextLine.startsWith("   ")) {
          // End of this result's metadata
          break;
        }

        const trimmed = nextLine.trim();
        if (trimmed.startsWith("Source:")) {
          url = trimmed.replace("Source:", "").trim();
        } else if (description === "" && trimmed.length > 0) {
          description = trimmed;
        }
      }

      if (title && url) {
        results.push({
          title,
          description,
          url,
        });
      }
      continue;
    }

    // Plain title fallback when backend sends unnumbered lines.
    const plainTitle = line.trim();
    if (plainTitle && !plainTitle.startsWith("Source:")) {
      let description = "";
      let url = "";

      for (let j = i + 1; j < lines.length; j++) {
        const nextLine = lines[j];
        if (!nextLine.startsWith("   ")) {
          break;
        }

        const trimmed = nextLine.trim();
        if (trimmed.startsWith("Source:")) {
          url = trimmed.replace("Source:", "").trim();
        } else if (description === "" && trimmed.length > 0) {
          description = trimmed;
        }
      }

      // Prevent system/meta lines from becoming numbered results.
      if (url) {
        results.push({
          title: plainTitle,
          description,
          url,
        });
      }
    }
  }

  const seen = new Set<string>();
  const deduped = results.filter((item) => {
    const key = extractRealUrl(item.url || "").trim().toLowerCase();
    if (!key) return true;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return deduped.length > 0 ? deduped : null;
}

function parseStructuredSearchPayload(text: string): StructuredSearchPayload | null {
  const raw = String(text || "").trim();
  if (!raw.startsWith("{")) return null;

  try {
    const parsed = JSON.parse(raw) as {
      message?: unknown;
      insights?: unknown;
      articles?: unknown;
    };
    if (!Array.isArray(parsed.articles)) {
      const onlyMessage = String(parsed.message ?? "").trim();
      if (!onlyMessage) return null;
      return { message: onlyMessage, insights: [], articles: [] };
    }

    const validArticles = parsed.articles
      .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
      .map((item) => ({
        title: String(item.title ?? "").trim(),
        description: String(item.description ?? "").trim(),
        url: String(item.url ?? "").trim(),
        source: String(item.source ?? "").trim(),
        timestamp: String(item.timestamp ?? "").trim(),
      }))
      .filter((article) => article.title && article.description && article.url && article.timestamp);

    const systemMessage = String(parsed.message ?? "").trim();
    const insights = Array.isArray(parsed.insights)
      ? parsed.insights.map((x) => String(x)).filter(Boolean)
      : [];

    if (validArticles.length === 0 && !systemMessage) return null;

    return { message: systemMessage, insights, articles: validArticles };
  } catch {
    return null;
  }
}

function parseInsightBullets(text: string): string[] {
  const lines = String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const cleaned = lines
    .map((line) => line.replace(/^[-*\d.)\s]+/, "").trim())
    .filter(Boolean);

  if (cleaned.length === 0) return [];
  if (cleaned.length === 1) return cleaned;
  return cleaned.slice(0, 6);
}

// ─── TYPES ────────────────────────────────────────────────────────────────────

type Role = "user" | "assistant" | "error";
type MessageKind = "text" | "image_analysis";

interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  loading: boolean;
  kind?: MessageKind;
  imagePreviewUrl?: string;
  imageAnalysis?: ImageAnalyzeResponse;
  responseComplete?: boolean;
  isRetryable?: boolean;
  retryPayload?: string;
}

interface ImageAnalysisPayload {
  objects: string[];
  text: string;
  description: string;
  possible_actions: string[];
}

interface ImageResource {
  title: string;
  link: string;
  summary: string;
}

interface ImageAnalyzeResponse {
  analysis: ImageAnalysisPayload;
  insights: string;
  resources: ImageResource[];
}

// ─── SUGGESTION CHIPS ─────────────────────────────────────────────────────────

const SUGGESTIONS = [
  {
    icon: ImageUp,
    label: "Analyze image",
    color: "text-orange-400",
    bg: "bg-orange-500/10 hover:bg-orange-500/20 border-orange-500/20 hover:border-orange-500/40",
  },
  {
    icon: Newspaper,
    label: "Search latest AI news",
    color: "text-blue-400",
    bg: "bg-blue-500/10 hover:bg-blue-500/20 border-blue-500/20 hover:border-blue-500/40",
  },
  {
    icon: TrendingUp,
    label: "Analyze stock trends",
    color: "text-emerald-400",
    bg: "bg-emerald-500/10 hover:bg-emerald-500/20 border-emerald-500/20 hover:border-emerald-500/40",
  },
  {
    icon: ListTodo,
    label: "Plan my tasks",
    color: "text-violet-400",
    bg: "bg-violet-500/10 hover:bg-violet-500/20 border-violet-500/20 hover:border-violet-500/40",
  },
  {
    icon: Search,
    label: "Summarize a topic",
    color: "text-amber-400",
    bg: "bg-amber-500/10 hover:bg-amber-500/20 border-amber-500/20 hover:border-amber-500/40",
  },
];

function SuggestionChips({ onSelect }: { onSelect: (s: string) => void }) {
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {SUGGESTIONS.map(({ icon: Icon, label, color, bg }) => (
        <button
          key={label}
          onClick={() => onSelect(label)}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl border text-sm font-medium transition-all duration-200 ${bg}`}
        >
          <Icon className={`w-3.5 h-3.5 ${color}`} />
          <span className="text-foreground/80">{label}</span>
        </button>
      ))}
    </div>
  );
}

// ─── HEADER ───────────────────────────────────────────────────────────────────

function Header({
  hasChatStarted,
  onActionSelect,
}: {
  hasChatStarted: boolean;
  onActionSelect: (action: "analyze_image" | "web_research" | "ask_question" | "summarize_content") => void;
}) {
  const [now] = useState(() => new Date());
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!panelRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) {
      document.addEventListener("mousedown", onDown);
    }
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const actionItems = [
    { key: "analyze_image" as const, label: "Analyze Image", icon: ImageUp },
    { key: "web_research" as const, label: "Web Research", icon: Search },
    { key: "ask_question" as const, label: "Ask Question", icon: Sparkles },
    { key: "summarize_content" as const, label: "Summarize Content", icon: ListTodo },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-4 sm:px-6 h-14 border-b border-border bg-background/80 backdrop-blur-xl">
      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <div className="relative w-8 h-8 rounded-lg bg-primary/15 border border-primary/25 flex items-center justify-center shadow-sm">
          <svg viewBox="0 0 32 32" className="w-4 h-4 text-primary" fill="currentColor">
            <path d="M16 2 L17.8 13 L27 10 L19.5 16 L27 22 L17.8 19 L16 30 L14.2 19 L5 22 L12.5 16 L5 10 L14.2 13 Z" />
          </svg>
          <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-emerald-400 rounded-full border-2 border-background shadow-sm" />
        </div>
        <div>
          <span className="font-semibold text-sm text-foreground tracking-tight">AgentPro</span>
          <div className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
            <span className="text-[10px] text-emerald-400 font-medium">Online</span>
          </div>
        </div>
      </div>

      {/* Right side */}
      <div className="relative flex items-center gap-2 sm:gap-3" ref={panelRef}>
        <span className="hidden sm:block text-xs text-muted-foreground">
          {formatHeaderDate(now)}
        </span>
        <button
          type="button"
          title="Explore agent capabilities"
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-secondary border border-border text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          <Sparkles className="w-3.5 h-3.5 text-primary" />
          <span>Explore</span>
          <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
        </button>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: -6 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: -6 }}
              transition={{ duration: 0.18 }}
              className="absolute right-0 top-11 z-[60] w-64 rounded-2xl border border-border bg-card/95 backdrop-blur-xl p-3 shadow-2xl"
            >
              <p className="text-sm font-semibold text-foreground mb-2">What do you want to do?</p>
              <div className="space-y-1.5">
                {actionItems.map(({ key, label, icon: Icon }) => (
                  <button
                    key={key}
                    onClick={() => {
                      setOpen(false);
                      onActionSelect(key);
                    }}
                    className="w-full flex items-center gap-2 rounded-xl px-2.5 py-2 text-sm text-foreground/85 hover:text-foreground hover:bg-accent transition-colors"
                  >
                    <Icon className="w-4 h-4 text-primary" />
                    <span>{label}</span>
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}

// ─── HERO ─────────────────────────────────────────────────────────────────────

function Hero() {
  const greeting = getGreeting();
  const emoji = getGreetingEmoji();

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="text-center space-y-2 pt-4"
    >
      <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-foreground">
        {greeting}, Saran {emoji}
      </h1>
      <p className="text-muted-foreground text-base">Ready to assist you today.</p>
    </motion.div>
  );
}

// ─── INTRO CARD ───────────────────────────────────────────────────────────────

function IntroCard({ onSuggest }: { onSuggest: (s: string) => void }) {
  const now = new Date();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
      className="w-full max-w-sm mx-auto"
    >
      <div className="bg-card rounded-2xl border border-border shadow-xl overflow-hidden">
        {/* Card header */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-border">
          <div className="w-8 h-8 rounded-lg bg-primary/15 border border-primary/20 flex items-center justify-center">
            <Bot className="w-4 h-4 text-primary" />
          </div>
          <div>
            <p className="text-sm font-semibold text-foreground">AI Agent</p>
            <p className="text-[11px] text-muted-foreground">{formatHeaderDate(now)}</p>
          </div>
          <div className="ml-auto">
            <Sparkles className="w-4 h-4 text-primary/60" />
          </div>
        </div>

      </div>
    </motion.div>
  );
}

// ─── SEARCH RESULTS CARD ──────────────────────────────────────────────────────

function SearchResultsCard({
  results,
  title,
  message,
  insights,
  requiredCount = 5,
}: {
  results: SearchResult[];
  title: string;
  message?: string;
  insights?: string[];
  requiredCount?: number;
}) {
  let counter = 1;
  console.log("Rendered items count:", results.length);
  return (
    <div className="space-y-3 w-full">
      <h3 className="text-sm font-semibold text-foreground/80">{title}</h3>
      {message && (
        <div className="rounded-lg border border-primary/30 bg-primary/10 px-3 py-2 text-xs text-foreground/85">
          {message}
        </div>
      )}
      {results.length < requiredCount && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          Fetching more relevant news...
        </div>
      )}
      {insights && insights.length > 0 && (
        <div className="rounded-lg border border-border/60 bg-card/60 px-3 py-2">
          <p className="text-xs font-semibold text-foreground/80 mb-1">Insights</p>
          <ul className="space-y-1">
            {insights.slice(0, 3).map((line, idx) => (
              <li key={`${line}-${idx}`} className="text-xs text-foreground/65">
                • {line}
              </li>
            ))}
          </ul>
        </div>
      )}
      {results.map((result, idx) => {
        const currentIndex = counter++;
        return (
        <motion.a
          key={`${extractRealUrl(result.url || "") || result.title}-${idx}`}
          href={extractRealUrl(result.url)}
          target="_blank"
          rel="noopener noreferrer"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: idx * 0.05 }}
          className="block p-3.5 rounded-xl bg-card/60 hover:bg-card border border-border/60 hover:border-border transition-all duration-200 group"
        >
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-semibold text-foreground/90 group-hover:text-primary transition-colors truncate">
                {currentIndex}. {result.title}
              </h4>
              {result.description && (
                <p className="text-xs text-foreground/60 mt-1.5 line-clamp-2">
                  {result.description}
                </p>
              )}
              {result.url && (
                <div className="mt-2.5 text-xs text-primary/70 group-hover:text-primary transition-colors">
                  🔗 {result.source || getDomain(result.url)}
                </div>
              )}
              {result.timestamp && (
                <div className="mt-1 text-[11px] text-muted-foreground">{result.timestamp}</div>
              )}
            </div>
          </div>
        </motion.a>
      )})}
    </div>
  );
}

// ─── MESSAGE CONTENT RENDERER ─────────────────────────────────────────────────

function renderContent(content: string) {
  const structuredSearch = parseStructuredSearchPayload(content);
  if (structuredSearch) {
    return (
      <SearchResultsCard
        results={structuredSearch.articles}
        title="Latest News Results"
        message={structuredSearch.message}
        insights={structuredSearch.insights}
        requiredCount={5}
      />
    );
  }

  // Check if this is a search results response
  const searchResults = parseSearchResults(content);
  if (searchResults) {
    // Extract title from first line
    const titleLine = content.split("\n")[0];
    const title = titleLine.replace("Top results for:", "").trim();
    return <SearchResultsCard results={searchResults} title={titleLine} />;
  }

  // Simple markdown-lite: bold (**text**), newlines
  const lines = content.split("\n");
  return lines.map((line, i) => {
    const parts = line.split(/(\*\*[^*]+\*\*)/g);
    return (
      <span key={i} className="block">
        {parts.map((part, j) =>
          part.startsWith("**") && part.endsWith("**") ? (
            <strong key={j} className="font-semibold text-foreground">
              {part.slice(2, -2)}
            </strong>
          ) : (
            <span key={j}>{part}</span>
          )
        )}
        {i < lines.length - 1 && <br />}
      </span>
    );
  });
}

function MessageBubble({
  message,
  onRetry,
}: {
  message: Message;
  onRetry?: (payload: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const isError = message.role === "error";
  const isLoading = message.loading;

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  if (isError) {
    const errorText =
      !message.content
        ? "Something went wrong. Please try again."
        : message.content;

    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex gap-3"
      >
        <div className="w-8 h-8 rounded-full bg-destructive/15 border border-destructive/30 flex-shrink-0 flex items-center justify-center mt-0.5">
          <AlertTriangle className="w-4 h-4 text-destructive" />
        </div>
        <div className="max-w-[80%] sm:max-w-[70%] bg-destructive/10 border border-destructive/25 rounded-2xl rounded-bl-sm px-4 py-3.5 space-y-2.5">
          <p className="text-sm font-semibold text-destructive">Something Went Wrong</p>
          <p className="text-sm text-foreground/80">{errorText}</p>
          {onRetry && message.retryPayload && (
            <button
              onClick={() => onRetry(message.retryPayload!)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-destructive/20 hover:bg-destructive/30 border border-destructive/30 text-destructive text-xs font-medium transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              Retry
            </button>
          )}
        </div>
      </motion.div>
    );
  }

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8, x: -10 }}
        animate={{ opacity: 1, y: 0, x: 0 }}
        transition={{ duration: 0.25 }}
        className="flex gap-3"
      >
        <div className="w-8 h-8 rounded-full bg-card border border-border flex-shrink-0 flex items-center justify-center shadow-sm">
          <Bot className="w-4 h-4 text-primary" />
        </div>
        <div className="bg-card border border-border px-4 py-3.5 rounded-2xl rounded-bl-sm flex items-center gap-1.5 shadow-sm min-h-[48px]">
          <span className="typing-dot text-muted-foreground" />
          <span className="typing-dot text-muted-foreground" />
          <span className="typing-dot text-muted-foreground" />
          {message.kind === "image_analysis" && (
            <span className="ml-2 text-xs text-muted-foreground">Analyzing image...</span>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8, x: isUser ? 10 : -10 }}
      animate={{ opacity: 1, y: 0, x: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={`flex gap-3 group ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center mt-0.5 shadow-sm ${
          isUser
            ? "bg-primary/20 border border-primary/30"
            : "bg-card border border-border"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-primary" />
        ) : (
          <Bot className="w-4 h-4 text-primary" />
        )}
      </div>

      {/* Content */}
      <div className={`flex flex-col gap-1 ${isUser ? "items-end max-w-[80%] sm:max-w-[70%]" : "items-start w-full max-w-[90%] sm:max-w-[75%]"}`}>
        {isUser ? (
          // User message in bubble
          <div className="px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm bg-primary text-primary-foreground rounded-br-sm max-w-full">
            <span className="whitespace-pre-wrap">{message.content}</span>
          </div>
        ) : (
          // Assistant image-analysis message
          message.kind === "image_analysis" && message.imageAnalysis ? (
            <div className="w-full rounded-2xl rounded-bl-sm border border-border bg-card p-3 sm:p-4 space-y-3">
              {message.imagePreviewUrl && (
                <div className="rounded-xl overflow-hidden border border-border/60 max-w-xs">
                  <img src={message.imagePreviewUrl} alt="Analyzed" className="w-full h-auto object-cover" />
                </div>
              )}

              <div>
                <p className="text-sm font-semibold text-foreground mb-1">Description</p>
                <p className="text-sm text-foreground/85">
                  {message.imageAnalysis.analysis.description || "No description detected."}
                </p>
              </div>

              {message.imageAnalysis.analysis.objects?.length > 0 && (
                <div>
                  <p className="text-sm font-semibold text-foreground mb-2">Detected Objects</p>
                  <div className="flex flex-wrap gap-2">
                    {message.imageAnalysis.analysis.objects.map((obj, idx) => (
                      <span
                        key={`${obj}-${idx}`}
                        className="px-2 py-1 rounded-md bg-secondary text-xs text-foreground/80 border border-border"
                      >
                        {obj}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <p className="text-sm font-semibold text-foreground mb-1">Insights</p>
                <ul className="space-y-1">
                  {parseInsightBullets(message.imageAnalysis.insights).map((point, idx) => (
                    <li key={`${point}-${idx}`} className="text-sm text-foreground/85">
                      • {point}
                    </li>
                  ))}
                </ul>
              </div>

              {message.imageAnalysis.resources?.length > 0 && (
                <div>
                  <p className="text-sm font-semibold text-foreground mb-2">Resources</p>
                  <div className="space-y-2">
                    {message.imageAnalysis.resources.map((item, idx) => (
                      <a
                        key={`${item.title}-${idx}`}
                        href={extractRealUrl(item.link)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block rounded-lg border border-border/70 p-2.5 hover:border-border hover:bg-card/60 transition-colors"
                      >
                        <p className="text-sm font-semibold text-foreground">{item.title || `Resource ${idx + 1}`}</p>
                        <p className="text-xs text-foreground/65 mt-1">{item.summary || "Open to learn more."}</p>
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) :
          // Assistant message - check if it's search results
          (parseStructuredSearchPayload(message.content) || parseSearchResults(message.content)) ? (
            // Search results use full width without bubble styling
            <div className="w-full">
              {renderContent(message.content)}
            </div>
          ) : (
            // Regular message in bubble
            <div className="px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm bg-card border border-border text-foreground rounded-bl-sm">
              <div className="whitespace-pre-wrap text-foreground/90">{renderContent(message.content)}</div>
            </div>
          )
        )}

        {/* Timestamp + copy */}
        <div className={`flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity ${isUser ? "flex-row-reverse" : ""}`}>
          <span className="text-[11px] text-muted-foreground">{formatTime(message.timestamp)}</span>
          {!isUser &&
            message.kind !== "image_analysis" &&
            !parseStructuredSearchPayload(message.content) &&
            !parseSearchResults(message.content) && (
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
            >
              {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// ─── TYPING INDICATOR ─────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10, y: 4 }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      exit={{ opacity: 0, x: -10 }}
      transition={{ duration: 0.25 }}
      className="flex gap-3"
    >
      <div className="w-8 h-8 rounded-full bg-card border border-border flex-shrink-0 flex items-center justify-center shadow-sm">
        <Bot className="w-4 h-4 text-primary" />
      </div>
      <div className="bg-card border border-border px-4 py-3.5 rounded-2xl rounded-bl-sm flex items-center gap-1.5 shadow-sm">
        <span className="typing-dot text-muted-foreground" />
        <span className="typing-dot text-muted-foreground" />
        <span className="typing-dot text-muted-foreground" />
      </div>
    </motion.div>
  );
}

// ─── INPUT BAR ────────────────────────────────────────────────────────────────

function InputBar({
  value,
  onChange,
  onSend,
  disabled,
  onUploadImage,
  onTakePhoto,
  onUploadFile,
  onConnectDrive,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  disabled: boolean;
  onUploadImage: () => void;
  onTakePhoto: () => void;
  onUploadFile: () => void;
  onConnectDrive: () => void;
  placeholder?: string;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!disabled && value.trim()) {
      onSend();
    }
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  };

  // Auto resize
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }, [value]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    if (menuOpen) {
      document.addEventListener("mousedown", onDown);
    }
    return () => document.removeEventListener("mousedown", onDown);
  }, [menuOpen]);

  return (
    <form onSubmit={handleSubmit} className="bg-card rounded-2xl border border-border shadow-xl overflow-visible transition-shadow focus-within:shadow-2xl focus-within:border-border/60">
      <textarea
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder={placeholder ?? "Message AgentPro..."}
        rows={1}
        disabled={disabled}
        className="w-full bg-transparent px-4 sm:px-5 pt-4 pb-2 text-foreground placeholder:text-muted-foreground resize-none outline-none text-sm leading-relaxed disabled:opacity-50 min-h-[52px]"
      />
      <div className="flex items-center justify-between px-4 sm:px-5 pb-3.5 pt-1 gap-3">
        {/* Left tools */}
        <div className="relative parent-container flex items-center gap-1.5" ref={menuRef}>
          <button
            type="button"
            title="Add input"
            onClick={() => setMenuOpen((v) => !v)}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          >
            <Plus className="w-4 h-4" />
          </button>
          <AnimatePresence>
            {menuOpen && (
              <motion.div
                initial={{ opacity: 0, scale: 0.96, y: 6 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96, y: 6 }}
                transition={{ duration: 0.16 }}
                className="popup-menu open"
              >
                <button
                  type="button"
                  onClick={() => {
                    setMenuOpen(false);
                    onUploadImage();
                  }}
                  className="popup-item"
                >
                  <ImageUp />
                  <span>Upload Image</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMenuOpen(false);
                    onTakePhoto();
                  }}
                  className="popup-item"
                >
                  <Camera />
                  <span>Take Photo</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMenuOpen(false);
                    onUploadFile();
                  }}
                  className="popup-item"
                >
                  <FileText />
                  <span>Upload File</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMenuOpen(false);
                    onConnectDrive();
                  }}
                  className="popup-item"
                >
                  <FileUp />
                  <span>Connect Drive</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>
          <button
            type="button"
            title="Voice"
            className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          >
            <Mic className="w-4 h-4" />
          </button>
        </div>

        {/* Right actions */}
        <div className="flex items-center gap-2">
          {disabled && (
            <span className="text-xs text-primary animate-pulse hidden sm:block">
              Agent is thinking…
            </span>
          )}
          {!disabled && value.trim() && (
            <span className="text-[11px] text-muted-foreground hidden sm:block">
              ↵ to send
            </span>
          )}
          <button
            type="submit"
            disabled={!value.trim() || disabled}
            className="w-8 h-8 rounded-xl bg-primary flex items-center justify-center text-primary-foreground disabled:opacity-25 disabled:cursor-not-allowed hover:opacity-90 active:scale-95 transition-all shadow-md"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </form>
  );
}

// ─── CHAT BOX (MESSAGES) ──────────────────────────────────────────────────────

function ChatBox({
  messages,
  onRetry,
}: {
  messages: Message[];
  onRetry: (payload: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto chat-scroll px-3 sm:px-0 py-4 space-y-4">
      <AnimatePresence initial={false}>
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} onRetry={onRetry} />
        ))}
      </AnimatePresence>
      <div ref={bottomRef} />
    </div>
  );
}

// ─── WELCOME VIEW ─────────────────────────────────────────────────────────────

function WelcomeView({ onSuggest }: { onSuggest: (s: string) => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-8 py-8 px-4">
      <Hero />
      <IntroCard onSuggest={onSuggest} />
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25, duration: 0.4 }}
        className="w-full max-w-2xl"
      >
        <SuggestionChips onSelect={onSuggest} />
      </motion.div>
    </div>
  );
}

// ─── CHAT HEADER BAR (active chat) ────────────────────────────────────────────

function ChatHeader({ messageCount, onClear }: { messageCount: number; onClear: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center justify-between py-2 border-b border-border mb-2"
    >
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">
          {messageCount} message{messageCount !== 1 ? "s" : ""}
        </span>
      </div>
      <button
        onClick={onClear}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded-lg hover:bg-accent"
      >
        <RefreshCw className="w-3 h-3" />
        New chat
      </button>
    </motion.div>
  );
}

function ImageAnalyzerCard({
  imagePreview,
  imageLoading,
  imageError,
  onPick,
  onAnalyze,
}: {
  imagePreview: string;
  imageLoading: boolean;
  imageError: string;
  onPick: (file: File | null) => void;
  onAnalyze: () => void;
}) {
  const [isDragActive, setIsDragActive] = useState(false);

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragActive(false);
    const file = e.dataTransfer.files?.[0] ?? null;
    onPick(file);
  };

  return (
    <div className="mb-3 rounded-2xl border border-border bg-card/60 p-3 sm:p-4 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-foreground">Image Understanding</p>
          <p className="text-xs text-muted-foreground">Upload a jpg/png image and get analysis + resources</p>
        </div>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragActive(true);
        }}
        onDragLeave={() => setIsDragActive(false)}
        onDrop={onDrop}
        className={`rounded-xl border border-dashed p-4 transition-colors ${
          isDragActive ? "border-primary bg-primary/5" : "border-border bg-background/40"
        }`}
      >
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <label className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-secondary hover:bg-accent cursor-pointer text-xs font-medium">
            <ImageUp className="w-3.5 h-3.5" />
            Choose image
            <input
              type="file"
              accept="image/jpeg,image/jpg,image/png"
              className="hidden"
              onChange={(e) => onPick(e.target.files?.[0] ?? null)}
            />
          </label>
          <p className="text-xs text-muted-foreground">or drag and drop here</p>
        </div>

        {imagePreview && (
          <div className="mt-3 rounded-xl overflow-hidden border border-border/60 max-w-xs">
            <img src={imagePreview} alt="Uploaded preview" className="w-full h-auto object-cover" />
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onAnalyze}
          disabled={!imagePreview || imageLoading}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {imageLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
          Analyze Image
        </button>
        {imageLoading && <span className="text-xs text-muted-foreground">Analyzing...</span>}
      </div>

      {imageError && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {imageError}
        </div>
      )}

    </div>
  );
}

// ─── ROOT PAGE ────────────────────────────────────────────────────────────────

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasChatStarted, setHasChatStarted] = useState(false);
  const [imagePreview, setImagePreview] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const inputAreaRef = useRef<HTMLDivElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const updateMessage = useCallback((id: string, patch: Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  const appendUniqueMessage = useCallback((next: Message) => {
    setMessages((prev) => {
      const alreadyExists = prev.some((m) => m.id === next.id);
      if (alreadyExists) return prev;

      const isDuplicatePayload = prev.some(
        (m) =>
          m.role === next.role &&
          m.kind === next.kind &&
          m.content.trim() === next.content.trim() &&
          Math.abs(m.timestamp.getTime() - next.timestamp.getTime()) < 1500
      );
      if (isDuplicatePayload) return prev;

      return [...prev, next];
    });
  }, []);

  const handleSend = useCallback(
    async (overrideInput?: string) => {
      const input = typeof overrideInput === "string" ? overrideInput : inputValue;
      let assistantId = "";

      try {
        if (!input || !input.trim() || isLoading) {
          console.log("Empty input");
          return;
        }

        if (!hasChatStarted) setHasChatStarted(true);
        setError("");

        const trimmedInput = input.trim();

        appendUniqueMessage({
          id: createMessageId(),
          role: "user",
          content: trimmedInput,
          timestamp: new Date(),
          loading: false,
          responseComplete: true,
        });
        setInputValue("");

        setIsLoading(true);
        assistantId = createMessageId();
        appendUniqueMessage({
          id: assistantId,
          role: "assistant",
          content: "",
          timestamp: new Date(),
          loading: true,
          responseComplete: false,
        });

        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: input.trim(),
          }),
        });

        console.log("Status:", res.status);

        if (!res.ok) {
          const errText = await res.text();
          console.error("Backend error:", errText);
          throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();
        console.log("Response:", data);

        const finalContent = String(
          data?.response ?? "I received your message but couldn't generate a response."
        );
        updateMessage(assistantId, {
          content: finalContent,
          loading: false,
          responseComplete: true,
        });
      } catch (err) {
        console.error("FINAL ERROR:", err);
        setError((err as Error).message);

        const fallbackInput = typeof overrideInput === "string" ? overrideInput : inputValue;
        if (assistantId) {
          updateMessage(assistantId, {
            role: "error",
            content: (err as Error).message,
            loading: false,
            responseComplete: true,
            isRetryable: true,
            retryPayload: fallbackInput,
          });
        }
      } finally {
        setIsLoading(false);
      }
    },
    [appendUniqueMessage, hasChatStarted, inputValue, isLoading, updateMessage]
  );

  const handleSuggest = useCallback(
    (suggestion: string) => {
      if (suggestion.toLowerCase().includes("analyze image")) {
        setHasChatStarted(true);
        setIsImageModalOpen(true);
        imageInputRef.current?.click();
        return;
      }
      void handleSend(suggestion);
      // Scroll input into view on mobile
      setTimeout(() => inputAreaRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }), 100);
    },
    [handleSend]
  );

  const handleImagePick = useCallback((file: File | null) => {
    if (!file) return;
    const allowed = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowed.includes(file.type)) {
      setImageError("Invalid file type. Please upload jpg, jpeg, or png.");
      return;
    }
    setImageError("");
    setImageFile(file);
    setIsImageModalOpen(true);

    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(String(reader.result || ""));
    };
    reader.onerror = () => setImageError("Failed to read image file.");
    reader.readAsDataURL(file);
  }, []);

  const handleActionSelect = useCallback(
    (action: "analyze_image" | "web_research" | "ask_question" | "summarize_content") => {
      setHasChatStarted(true);
      if (action === "analyze_image") {
        setIsImageModalOpen(true);
        imageInputRef.current?.click();
        return;
      }
      if (action === "web_research") {
        void handleSend("Search latest AI news and summarize the key updates");
        return;
      }
      if (action === "ask_question") {
        setInputValue("I have a question about ");
        inputAreaRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
        return;
      }
      setInputValue("Summarize this content: ");
      inputAreaRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    },
    [handleSend]
  );

  const handleUploadImage = useCallback(() => {
    setHasChatStarted(true);
    setIsImageModalOpen(true);
    imageInputRef.current?.click();
  }, []);

  const handleTakePhoto = useCallback(() => {
    setHasChatStarted(true);
    setIsImageModalOpen(true);
    cameraInputRef.current?.click();
  }, []);

  const handleUploadFile = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleConnectDrive = useCallback(() => {
    setInputValue("Connect Drive integration coming soon. I can still help if you paste links or text.");
  }, []);

  const handleGenericFilePick = useCallback(
    (file: File | null) => {
      if (!file) return;
      const isImage = file.type.startsWith("image/");
      if (isImage) {
        handleImagePick(file);
        setHasChatStarted(true);
        setIsImageModalOpen(true);
        return;
      }
      const hint = `Uploaded file: ${file.name}. Ask me to summarize or explain this file content.`;
      setInputValue(hint);
    },
    [handleImagePick]
  );

  const handleAnalyzeImage = useCallback(async () => {
    if (!imageFile || imageLoading) return;

    setImageLoading(true);
    setImageError("");
    const analysisMsgId = createMessageId();
    const previewForMessage = imagePreview;

    appendUniqueMessage({
      id: analysisMsgId,
      role: "assistant",
      kind: "image_analysis",
      content: "Analyzing image...",
      timestamp: new Date(),
      loading: true,
      responseComplete: false,
    });

    try {
      const formData = new FormData();
      formData.append("file", imageFile);

      const res = await fetch(apiUrl("/analyze-image"), {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("Image API non-200:", res.status, errorText);
        throw new Error(`HTTP ${res.status}: ${errorText}`);
      }

      const data = (await res.json()) as ImageAnalyzeResponse;
      updateMessage(analysisMsgId, {
        content: "Image analysis complete.",
        loading: false,
        responseComplete: true,
        imagePreviewUrl: previewForMessage,
        imageAnalysis: data,
        kind: "image_analysis",
      });
      setIsImageModalOpen(false);
      if (!hasChatStarted) {
        setHasChatStarted(true);
      }
    } catch (err) {
      console.error("Image API Error:", err);
      const message = err instanceof Error ? err.message : "Image analysis failed";
      setImageError(message);
      updateMessage(analysisMsgId, {
        role: "error",
        content: "Unable to analyze image right now. Please try again later.",
        loading: false,
        responseComplete: true,
        isRetryable: false,
      });
    } finally {
      setImageLoading(false);
    }
  }, [imageFile, imageLoading, hasChatStarted, imagePreview, appendUniqueMessage, updateMessage]);

  const handleClear = () => {
    setMessages([]);
    setHasChatStarted(false);
    setInputValue("");
    setIsLoading(false);
    setImageError("");
    setImagePreview("");
    setImageFile(null);
    setIsImageModalOpen(false);
    setError("");
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Header hasChatStarted={hasChatStarted} onActionSelect={handleActionSelect} />

      {/* Main content */}
      <main className="flex-1 min-h-0 flex flex-col pt-14">
        <div className="flex-1 min-h-0 flex flex-col w-full max-w-2xl mx-auto px-3 sm:px-4">
          <AnimatePresence mode="wait">
            {!hasChatStarted ? (
              <motion.div
                key="welcome"
                className="flex-1 min-h-0 flex flex-col"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                <WelcomeView onSuggest={handleSuggest} />
              </motion.div>
            ) : (
              <motion.div
                key="chat"
                className="flex-1 min-h-0 flex flex-col"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                <ChatHeader messageCount={messages.length} onClear={handleClear} />
                <ChatBox messages={messages} onRetry={(payload) => void handleSend(payload)} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Input area — always visible */}
          <div ref={inputAreaRef} className="flex-shrink-0 pb-4 pt-2">
            <InputBar
              value={inputValue}
              onChange={setInputValue}
              onSend={() => void handleSend()}
              disabled={isLoading}
              onUploadImage={handleUploadImage}
              onTakePhoto={handleTakePhoto}
              onUploadFile={handleUploadFile}
              onConnectDrive={handleConnectDrive}
            />
            {error && (
              <p className="mt-2 text-center text-xs text-destructive">{error}</p>
            )}
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleImagePick(e.target.files?.[0] ?? null)}
            />
            <input
              ref={cameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              className="hidden"
              onChange={(e) => handleImagePick(e.target.files?.[0] ?? null)}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.pdf,.txt"
              className="hidden"
              onChange={(e) => handleGenericFilePick(e.target.files?.[0] ?? null)}
            />
            <p className="text-center text-xs text-muted-foreground mt-2.5">
              Ask anything. I will handle the rest.
            </p>
          </div>
        </div>
      </main>

      <AnimatePresence>
        {isImageModalOpen && (
          <motion.div
            className="image-modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsImageModalOpen(false)}
          >
            <motion.div
              className="image-modal"
              initial={{ opacity: 0, scale: 0.96, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 8 }}
              transition={{ duration: 0.18 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-sm font-semibold text-foreground">Image Analysis</p>
                  <p className="text-xs text-muted-foreground">Upload and analyze an image on demand</p>
                </div>
                <button
                  type="button"
                  onClick={() => setIsImageModalOpen(false)}
                  className="w-7 h-7 rounded-lg text-muted-foreground hover:text-foreground hover:bg-accent inline-flex items-center justify-center"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <ImageAnalyzerCard
                imagePreview={imagePreview}
                imageLoading={imageLoading}
                imageError={imageError}
                onPick={handleImagePick}
                onAnalyze={handleAnalyzeImage}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
