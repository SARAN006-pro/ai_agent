"""
Tool implementations.

Each tool is self-contained: it handles its own errors and never
raises an exception to the caller. Failed tools return a ToolResult
with success=False and a descriptive error, so the agent can reason
about what went wrong and try a different approach.

Tools here simulate real APIs (no live internet access in this
environment), but the structure mirrors exactly how you'd build
tools against real endpoints.
"""

import math
import os
import json
import re
import ast
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlparse, unquote
from urllib.request import Request, urlopen
from tools import BaseTool, ToolResult
from logger import AgentLogger


# ─────────────────────────────────────────────
# Tool 1: Web Search
# ─────────────────────────────────────────────

_SEARCH_DB = {
    "anthropic claude": {
        "title": "Anthropic Claude — AI Assistant",
        "summary": (
            "Claude is Anthropic's AI assistant. The latest models include "
            "Claude Opus 4.5 (most capable), Claude Sonnet 4.5 (balanced), "
            "and Claude Haiku 4.5 (fastest). Claude supports tool use, vision, "
            "and long context windows up to 200K tokens. Anthropic's focus is "
            "on Constitutional AI and safety."
        ),
        "source": "anthropic.com",
    },
    "python agentic ai": {
        "title": "Building Agentic AI with Python",
        "summary": (
            "Agentic AI refers to systems where an LLM autonomously selects "
            "and executes tools in a loop to complete tasks. Key patterns: "
            "ReAct (Reason+Act), Plan-and-Execute, and Reflexion. Popular "
            "frameworks include LangChain, CrewAI, and AutoGen. Core challenge "
            "is reliability: agents must handle tool failures and avoid loops."
        ),
        "source": "arxiv.org",
    },
    "react prompting": {
        "title": "ReAct: Synergizing Reasoning and Acting in LLMs",
        "summary": (
            "ReAct (Yao et al., 2022) is a prompting strategy where the model "
            "interleaves reasoning traces (Thought) with actions (Act) and "
            "observations (Obs). This improves grounding and reduces "
            "hallucination compared to pure chain-of-thought. The loop "
            "continues until the model produces a final answer."
        ),
        "source": "arxiv.org/abs/2210.03629",
    },
    "langchain": {
        "title": "LangChain — LLM Application Framework",
        "summary": (
            "LangChain is a Python/JS framework for building LLM applications. "
            "Key abstractions: chains, agents, memory, and retrievers. "
            "AgentExecutor runs the tool-calling loop. Pros: fast prototyping, "
            "large ecosystem. Cons: heavy abstraction can obscure what's "
            "happening and make debugging hard. Best for complex pipelines."
        ),
        "source": "python.langchain.com",
    },
    "goklouds ai intern": {
        "title": "GoKloud AI Intern Role",
        "summary": (
            "GoKloud is hiring AI Software Engineer interns focused on agentic "
            "workflows. Requirements: Python, Claude API, experience with AI "
            "agents, chain-of-thought prompting. The role involves building "
            "autonomous agents for real product integration."
        ),
        "source": "goklouds.com/careers",
    },
}

class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for information on a topic. "
        "Use this to find facts, explanations, or background knowledge. "
        "Returns a title, summary, and source URL."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            }
        },
        "required": ["query"],
    }

    def __init__(self):
        self.log = AgentLogger("WebSearchTool")
        self.timeout = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "8"))
        self.max_query_attempts = int(os.getenv("WEB_SEARCH_MAX_QUERY_ATTEMPTS", "3"))
        self.required_news_count = int(os.getenv("NEWS_REQUIRED_COUNT", "5"))
        self.min_novelty_ratio = float(os.getenv("NEWS_MIN_NOVELTY_RATIO", "0.7"))
        self._news_state_lock = threading.Lock()
        self._news_history: dict[str, dict] = {}
        self._angle_cycle = [
            "general_headlines",
            "startups_funding",
            "research_breakthroughs",
            "big_tech_moves",
            "policy_regulation",
        ]

    def _news_topic_key(self, query: str) -> str:
        cleaned = self._clean_query(query)
        cleaned = re.sub(
            r"\b(news|today|latest|breaking|updates|headlines|again|more|another|different|else|next|also)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or self._clean_query(query) or "general"

    def _has_follow_up_cue(self, query: str) -> bool:
        q = (query or "").lower()
        return any(token in q for token in ("again", "more", "another", "different", "next", "also", "else"))

    def _angle_terms(self, angle: str) -> str:
        mapping = {
            "general_headlines": "top headlines updates",
            "startups_funding": "startup funding venture capital deals",
            "research_breakthroughs": "research paper breakthrough benchmark",
            "big_tech_moves": "OpenAI Google Meta Microsoft announcements",
            "policy_regulation": "policy regulation law government safety",
        }
        return mapping.get(angle, "top headlines updates")

    def _select_news_angle(self, topic_key: str, query: str) -> tuple[str, str]:
        q = (query or "").lower()
        explicit_map = {
            "funding": "startups_funding",
            "startup": "startups_funding",
            "research": "research_breakthroughs",
            "paper": "research_breakthroughs",
            "breakthrough": "research_breakthroughs",
            "openai": "big_tech_moves",
            "google": "big_tech_moves",
            "meta": "big_tech_moves",
            "microsoft": "big_tech_moves",
            "policy": "policy_regulation",
            "regulation": "policy_regulation",
            "law": "policy_regulation",
        }
        for token, angle in explicit_map.items():
            if token in q:
                return angle, "follow_up" if self._has_follow_up_cue(query) else "new_topic"

        with self._news_state_lock:
            state = self._news_history.get(topic_key) or {
                "urls": set(),
                "sources": set(),
                "angles": [],
                "requests": 0,
            }
            prior_requests = int(state.get("requests", 0))

        if prior_requests == 0:
            return self._angle_cycle[0], "new_topic"

        angle_idx = prior_requests % len(self._angle_cycle)
        intent = "repeat_request"
        if self._has_follow_up_cue(query):
            intent = "follow_up"
        return self._angle_cycle[angle_idx], intent

    def _rank_news_results_by_novelty(self, topic_key: str, items: list[dict]) -> tuple[list[dict], float, int]:
        with self._news_state_lock:
            state = self._news_history.get(topic_key) or {
                "urls": set(),
                "sources": set(),
                "angles": [],
                "requests": 0,
            }
            seen_urls = set(state.get("urls", set()))

        unseen: list[dict] = []
        seen: list[dict] = []
        for item in items:
            key = self._canonicalize_url(str(item.get("url", "")))
            if key and key in seen_urls:
                seen.append(item)
            else:
                unseen.append(item)

        required = max(self.required_news_count, 1)
        selected = (unseen + seen)[:required]
        novel_count = min(len(unseen), len(selected))
        novelty_ratio = (novel_count / len(selected)) if selected else 1.0
        return selected, novelty_ratio, novel_count

    def _update_news_history(self, topic_key: str, angle: str, selected: list[dict]) -> None:
        with self._news_state_lock:
            state = self._news_history.get(topic_key)
            if not state:
                state = {"urls": set(), "sources": set(), "angles": [], "requests": 0}
                self._news_history[topic_key] = state

            state["requests"] = int(state.get("requests", 0)) + 1
            state.setdefault("angles", []).append(angle)

            url_set = state.setdefault("urls", set())
            source_set = state.setdefault("sources", set())
            for item in selected:
                url_key = self._canonicalize_url(str(item.get("url", "")))
                if url_key:
                    url_set.add(url_key)
                source = str(item.get("source", "")).strip().lower()
                if source:
                    source_set.add(source)

    def _canonicalize_url(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        # Normalize DuckDuckGo redirect links to their target URL when present.
        m = re.search(r"[?&]uddg=([^&]+)", raw)
        if m:
            try:
                raw = unquote(m.group(1))
            except Exception:
                pass
        raw = re.sub(r"^https?://", "", raw, flags=re.IGNORECASE)
        return raw.rstrip("/").lower()

    def _is_news_mode(self, query: str) -> bool:
        q = (query or "").lower()
        return any(token in q for token in ("news", "latest", "today"))

    def _clean_query(self, query: str) -> str:
        cleaned = (query or "").lower()
        remove_words = [
            "search",
            "give",
            "then",
            "show",
            "tell me",
            "find",
            "latest",
            "news about",
            "again",
            "more",
            "another",
            "different",
            "else",
            "next",
            "also",
        ]
        for word in remove_words:
            cleaned = cleaned.replace(word, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _news_query_variants(self, query: str, angle_terms: str = "") -> list[str]:
        cleaned_query = self._clean_query(query)
        base = re.sub(r"\s+", " ", cleaned_query or (query or "").strip())
        contextual_base = f"{base} {angle_terms}".strip() if angle_terms else base
        candidates = [
            contextual_base,
            f"{contextual_base} latest updates",
            f"{contextual_base} breaking news",
            f"{contextual_base} site:reuters.com OR site:apnews.com OR site:bbc.com OR site:cnn.com",
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            key = c.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(c.strip())
        return deduped[:8]

    def _extract_domain(self, url: str) -> str:
        try:
            real = url
            if "uddg=" in real:
                m = re.search(r"[?&]uddg=([^&]+)", real)
                if m:
                    real = unquote(m.group(1))
            parsed = urlparse(real if real.startswith("http") else f"https://{real}")
            return parsed.netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def _is_generic_landing_page(self, url: str) -> bool:
        lowered = (url or "").lower()
        if any(p in lowered for p in ("/category", "/topic", "/home", "/section", "/tag/")):
            return True
        try:
            parsed = urlparse(url if url.startswith("http") else f"https://{url}")
            path = (parsed.path or "").strip("/")
            if path in ("", "home", "news", "latest", "topics", "category"):
                return True
            if parsed.query and "q=" in parsed.query and "news" in parsed.query and len(path) <= 1:
                return True
        except Exception:
            return False
        return False

    def _is_article_like_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url if url.startswith("http") else f"https://{url}")
            path = (parsed.path or "").strip("/")
            if not path:
                return False
            segments = [s for s in path.split("/") if s]
            if len(segments) < 2:
                return False
            if any(seg in {"category", "topic", "topics", "home", "section", "tag"} for seg in segments):
                return False
            # Article pages usually include a descriptive slug or date-based path.
            return bool(re.search(r"\d{4}/\d{2}/\d{2}|[a-z0-9-]{12,}", path.lower()))
        except Exception:
            return False

    def _looks_recent(self, title: str, snippet: str, url: str) -> bool:
        hay = f"{title} {snippet} {url}".lower()
        if re.search(r"\b(today|just now|minutes? ago|hours? ago|breaking|live|updated)\b", hay):
            return True
        # Keep current-year signals as recent heuristics for providers without explicit timestamp fields.
        current_year = datetime.now(timezone.utc).year
        if str(current_year) in hay:
            return True
        return False

    def _is_irrelevant_for_news(self, title: str, snippet: str, url: str) -> bool:
        hay = f"{title} {snippet} {url}".lower()
        blocked_terms = (
            "arxiv",
            "tutorial",
            "how to",
            "course",
            "documentation",
            "wikipedia",
            "blog",
            "opinion",
        )
        return any(term in hay for term in blocked_terms)

    def _light_filter(self, items: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for item in items:
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("description", item.get("snippet", ""))).strip()
            if len(title.split()) >= 4 or "news" in snippet.lower():
                filtered.append(item)
        return filtered

    def _filter_news_results(self, items: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for item in self._dedupe_results(items):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            url = str(item.get("url", "")).strip()
            source = str(item.get("source", "")).strip() or self._extract_domain(url)

            if not title or not snippet or not url:
                continue

            normalized.append(
                {
                    "title": title,
                    "description": snippet,
                    "url": url,
                    "source": source,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        filtered = self._light_filter(normalized)
        print("RAW RESULTS:", len(normalized))
        print("FILTERED RESULTS:", len(filtered))

        if not filtered and normalized:
            filtered = normalized[:3]

        return filtered

    def _search_variant_results(self, variant: str) -> list[dict]:
        collected: list[dict] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self._duckduckgo_api_search, variant): "duckduckgo_api",
                pool.submit(self._duckduckgo_html_search, variant): "duckduckgo_html",
            }
            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    result_items = future.result()
                    if result_items:
                        self.log.info(
                            f"news candidate strategy={strategy} query='{variant}' count={len(result_items)}"
                        )
                        collected.extend(result_items)
                except (HTTPError, URLError, TimeoutError) as e:
                    self.log.warn(f"{strategy} error: {e}")
                except Exception as e:
                    self.log.warn(f"{strategy} unexpected error: {e}")
        return self._dedupe_results(collected)

    def _run_news_search(self, query: str) -> ToolResult:
        required = max(self.required_news_count, 1)
        topic_key = self._news_topic_key(query)
        angle, intent_state = self._select_news_angle(topic_key, query)
        angle_terms = self._angle_terms(angle)
        variants = self._news_query_variants(query, angle_terms=angle_terms)
        gathered: list[dict] = []
        seen_urls: set[str] = set()

        def append_unique(items: list[dict]) -> None:
            for article in items:
                key = self._canonicalize_url(str(article.get("url", "")))
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                gathered.append(article)

        # Single pass with light filtering and novelty-aware ranking.
        for variant in variants:
            candidates = self._search_variant_results(variant)
            append_unique(self._filter_news_results(candidates))
            if len(gathered) >= required:
                break

        selected, novelty_ratio, novel_count = self._rank_news_results_by_novelty(topic_key, gathered)

        # Exploration trigger for repeat/follow-up requests when novelty is low.
        if selected and novelty_ratio < self.min_novelty_ratio and intent_state in {"repeat_request", "follow_up"}:
            alt_angle_index = (self._angle_cycle.index(angle) + 1) % len(self._angle_cycle)
            alt_angle = self._angle_cycle[alt_angle_index]
            alt_variants = self._news_query_variants(query, angle_terms=self._angle_terms(alt_angle))
            for variant in alt_variants:
                candidates = self._search_variant_results(variant)
                append_unique(self._filter_news_results(candidates))
                if len(gathered) >= required * 2:
                    break
            selected, novelty_ratio, novel_count = self._rank_news_results_by_novelty(topic_key, gathered)
            angle = f"{angle}->{alt_angle}"

        if not selected and gathered:
            selected = gathered[:required]
            novelty_ratio = 1.0
            novel_count = len(selected)

        self._update_news_history(topic_key, angle, selected)

        articles = selected[:required]
        status = "ok" if len(articles) >= required else ("no_results" if len(articles) == 0 else "partial")
        message = ""
        if status == "partial":
            message = "Returning best available diverse news set; more results can be fetched."
        elif status == "no_results":
            message = "No matching news results found for this query."

        insights = [
            f"News mode enabled for query: {query}",
            f"Intent state: {intent_state}",
            f"Diversity angle: {angle}",
            f"Novelty ratio: {novelty_ratio:.2f}",
            "Query is cleaned before running search variants.",
            "Light filtering is applied to avoid over-filtering.",
        ]

        output = {
            "status": status,
            "mode": "news",
            "required_count": required,
            "query": query,
            "topic_key": topic_key,
            "intent_state": intent_state,
            "angle": angle,
            "novelty_ratio": round(novelty_ratio, 3),
            "novel_count": novel_count,
            "message": message,
            "insights": insights,
            "articles": articles,
            # Legacy compatibility for existing render path.
            "results": [
                {
                    "title": a["title"],
                    "snippet": a["description"],
                    "url": a["url"],
                    "source": a["source"],
                    "timestamp": a["timestamp"],
                }
                for a in articles
            ],
        }
        return ToolResult(success=True, output=output)

    def _dedupe_results(self, items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = self._canonicalize_url(str(item.get("url", "")))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(item)
        return deduped

    def _build_query_variants(self, query: str) -> list[str]:
        """Generate simple rewrite candidates for resilient search."""
        base = re.sub(r"\s+", " ", query).strip()
        variants = [base]
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", base).strip()
        if cleaned and cleaned.lower() != base.lower():
            variants.append(cleaned)
        if not re.search(r"\b(latest|today|202[0-9]|2025|2026)\b", cleaned.lower()):
            variants.append(f"{cleaned} latest")

        deduped: list[str] = []
        seen: set[str] = set()
        for v in variants:
            key = v.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(v.strip())
        return deduped[: max(self.max_query_attempts, 1)]

    def _duckduckgo_api_search(self, query: str) -> list[dict]:
        """Use DuckDuckGo Instant Answer API as the first external source."""
        url = (
            "https://api.duckduckgo.com/?q="
            f"{quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        )
        req = Request(url, headers={"User-Agent": "agent-system-openrouter/1.0"})
        with urlopen(req, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")

        try:
            data = json.loads(payload)
        except Exception:
            return []

        results: list[dict] = []
        abstract = (data.get("AbstractText") or "").strip()
        abstract_source = (data.get("AbstractSource") or "DuckDuckGo").strip()
        abstract_url = (data.get("AbstractURL") or "").strip()
        heading = (data.get("Heading") or query).strip()
        if abstract:
            results.append(
                {
                    "title": heading,
                    "snippet": abstract,
                    "url": abstract_url or None,
                    "source": abstract_source,
                }
            )

        related = data.get("RelatedTopics") or []
        for topic in related:
            if len(results) >= 5:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    {
                        "title": (topic.get("Text") or "").split(" - ")[0][:120],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL"),
                        "source": "DuckDuckGo",
                    }
                )

        return self._dedupe_results(results)

    def _duckduckgo_html_search(self, query: str) -> list[dict]:
        """Fallback HTML scraping when the instant-answer endpoint is sparse."""
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=self.timeout) as response:
            html = response.read().decode("utf-8", errors="replace")

        title_matches = re.findall(
            r'<a[^>]*class="[^\"]*result__a[^\"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet_matches = re.findall(
            r'<a[^>]*class="[^\"]*result__snippet[^\"]*"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results: list[dict] = []
        for idx, (href, raw_title) in enumerate(title_matches[:5]):
            title = re.sub(r"<.*?>", "", raw_title).strip()
            snippet = ""
            if idx < len(snippet_matches):
                snippet = re.sub(r"<.*?>", "", snippet_matches[idx]).strip()
            results.append(
                {
                    "title": title or f"Result {idx+1}",
                    "snippet": snippet or "No snippet returned by source page.",
                    "url": href,
                    "source": "DuckDuckGo",
                }
            )
        return self._dedupe_results(results)

    def _local_search(self, query: str) -> list[dict]:
        """Deterministic local fallback for offline or blocked network scenarios."""
        query_lower = query.lower().strip()
        best_key = None
        best_score = 0
        for key in _SEARCH_DB:
            key_words = set(key.split())
            query_words = set(query_lower.split())
            score = len(key_words & query_words)
            if score > best_score:
                best_score = score
                best_key = key

        if not best_key or best_score == 0:
            return []

        result = _SEARCH_DB[best_key]
        return [
            {
                "title": result["title"],
                "snippet": result["summary"],
                "url": result["source"],
                "source": "local_fallback",
            }
        ]

    def run(self, query: str = "") -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Search query is empty. Provide a non-empty query string.",
            )

        search_query = self._clean_query(query) or query.strip()

        if self._is_news_mode(query):
            self.log.info(f"mode=news query='{query}' cleaned='{search_query}'")
            return self._run_news_search(search_query)

        variants = self._build_query_variants(search_query)
        last_error = None

        for attempt_index, variant in enumerate(variants, start=1):
            self.log.info(f"search attempt {attempt_index}/{len(variants)} query='{variant}'")

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(self._duckduckgo_api_search, variant): "duckduckgo_api",
                    pool.submit(self._duckduckgo_html_search, variant): "duckduckgo_html",
                }
                for future in as_completed(futures):
                    strategy = futures[future]
                    try:
                        results = future.result()
                        if results:
                            deduped_results = self._dedupe_results(results)
                            return ToolResult(
                                success=True,
                                output={
                                    "status": "ok",
                                    "query": query,
                                    "cleaned_query": search_query,
                                    "used_query": variant,
                                    "strategy": strategy,
                                    "results": deduped_results,
                                },
                            )
                    except (HTTPError, URLError, TimeoutError) as e:
                        last_error = f"{strategy} error: {e}"
                        self.log.warn(last_error)
                    except Exception as e:
                        last_error = f"{strategy} unexpected error: {e}"
                        self.log.warn(last_error)

        local_results = self._local_search(search_query)
        if local_results:
            return ToolResult(
                success=True,
                output={
                    "status": "fallback_local",
                    "query": query,
                    "cleaned_query": search_query,
                    "used_query": search_query,
                    "strategy": "local_fallback",
                    "results": local_results,
                },
            )

        try:
            return ToolResult(
                success=False,
                error=(
                    "Web search returned no results after retries and fallbacks. "
                    f"Original query='{query}'. Last error='{last_error or 'none'}'. "
                    "Try a more specific query with named entities or timeframe."
                ),
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {e}")


class QueryRefinerTool(BaseTool):
    name = "refine_query"
    description = (
        "Refine a vague user query into a focused search query with alternatives. "
        "Use this before web_search when the task is ambiguous, broad, or missing entities."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "raw_query": {
                "type": "string",
                "description": "The original user query that needs refinement.",
            },
            "intent": {
                "type": "string",
                "description": "Optional intent hint, e.g. latest news, definition, comparison, tutorial.",
            },
        },
        "required": ["raw_query"],
    }

    def run(self, raw_query: str = "", intent: str = "") -> ToolResult:
        try:
            if not raw_query or not raw_query.strip():
                return ToolResult(success=False, error="raw_query is empty.")

            base = re.sub(r"\s+", " ", raw_query).strip()
            base_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", base)
            tokens = [t for t in base_clean.split() if len(t) > 2]
            tokens = tokens[:8]

            if intent:
                refined = f"{base} {intent}".strip()
            else:
                refined = " ".join(tokens) if tokens else base

            alternatives = [
                refined,
                f"{refined} latest",
                f"{refined} official source",
            ]
            alternatives = [a.strip() for a in alternatives if a.strip()]

            # Preserve order while removing duplicates.
            seen: set[str] = set()
            unique_alts: list[str] = []
            for a in alternatives:
                key = a.lower()
                if key not in seen:
                    seen.add(key)
                    unique_alts.append(a)

            return ToolResult(
                success=True,
                output={
                    "raw_query": raw_query,
                    "intent": intent or None,
                    "refined_query": unique_alts[0],
                    "alternative_queries": unique_alts[1:3],
                    "notes": "Use refined_query first, then alternative_queries if search fails.",
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Query refinement failed: {e}")


# ─────────────────────────────────────────────
# Tool 2: Calculator
# ─────────────────────────────────────────────

class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Supports arithmetic, "
        "percentages, powers, sqrt, and basic trigonometry. "
        "Use for any numerical computation rather than estimating."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate, e.g. '(120 * 0.15) + 45'",
            }
        },
        "required": ["expression"],
    }

    # Whitelist safe names for eval — never eval arbitrary code
    _SAFE_NAMES = {
        "sqrt": math.sqrt, "pow": math.pow, "abs": abs,
        "round": round, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi, "e": math.e,
        "log": math.log, "log10": math.log10, "ceil": math.ceil,
        "floor": math.floor,
    }

    _ALLOWED_BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
        ast.Mod: lambda a, b: a % b,
    }

    _ALLOWED_UNARYOPS = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def _eval_node(self, node):
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant type")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_BINOPS:
                raise ValueError("Unsupported operator")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._ALLOWED_BINOPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._ALLOWED_UNARYOPS:
                raise ValueError("Unsupported unary operator")
            operand = self._eval_node(node.operand)
            return self._ALLOWED_UNARYOPS[op_type](operand)
        if isinstance(node, ast.Name):
            if node.id in self._SAFE_NAMES:
                return self._SAFE_NAMES[node.id]
            raise ValueError(f"Unknown identifier: {node.id}")
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            if func not in self._SAFE_NAMES.values():
                raise ValueError("Unsafe function call")
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)
        raise ValueError("Unsupported expression")

    def run(self, expression: str = "") -> ToolResult:
        try:
            # Strip anything that isn't a safe math expression
            cleaned = re.sub(r"[^0-9+\-*/().,%^ a-z_]", "", expression.lower())
            cleaned = cleaned.replace("^", "**").replace("%", "/100")
            if not cleaned.strip():
                return ToolResult(success=False, error="Empty expression.")

            node = ast.parse(cleaned, mode="eval")
            result = self._eval_node(node)

            return ToolResult(
                success=True,
                output={
                    "expression": expression,
                    "result": round(result, 6) if isinstance(result, float) else result,
                }
            )
        except ZeroDivisionError:
            return ToolResult(success=False, error="Division by zero.")
        except Exception as e:
            return ToolResult(success=False, error=f"Could not evaluate '{expression}': {e}")


# ─────────────────────────────────────────────
# Tool 3: Concept Explainer
# ─────────────────────────────────────────────

_CONCEPTS = {
    "react": (
        "ReAct (Reason + Act) is a prompting pattern where an LLM alternates "
        "between reasoning about a problem and taking an action (e.g. calling a tool). "
        "Each action produces an observation, which informs the next reasoning step. "
        "This loop continues until the model produces a final answer. ReAct reduces "
        "hallucination because the model grounds its reasoning in real tool outputs "
        "rather than relying purely on its training data."
    ),
    "chain of thought": (
        "Chain-of-thought (CoT) prompting asks the model to think step by step before "
        "producing a final answer. By explicitly laying out intermediate reasoning, "
        "the model allocates more compute to complex sub-problems. CoT is simple to "
        "implement (just add 'think step by step' to the prompt) and significantly "
        "improves accuracy on multi-step tasks like math, logic, and planning."
    ),
    "tool calling": (
        "Tool calling (also called function calling) is a mechanism where an LLM can "
        "request execution of an external function. The model is given tool schemas "
        "(name, description, input spec). When it decides to use one, it returns a "
        "structured call instead of plain text. The host application executes the tool "
        "and returns the result. This grounds the LLM in real-world data and actions."
    ),
    "self reflection": (
        "Self-reflection is a multi-call pattern where an LLM critiques its own output "
        "and then improves it. Typical flow: (1) Generate a draft. (2) Ask the model "
        "to identify weaknesses in the draft. (3) Ask the model to rewrite addressing "
        "those weaknesses. This consistently produces higher quality outputs than "
        "single-call generation, at the cost of 3x the API calls."
    ),
    "agentic workflow": (
        "An agentic workflow is a structured sequence of AI-driven steps where each "
        "step can involve LLM reasoning, tool calls, and branching. Unlike a single "
        "prompt, an agentic workflow can adapt its path based on intermediate results. "
        "Examples: a research pipeline that searches → summarizes → identifies gaps → "
        "searches again. The key property is that the model controls the execution path."
    ),
}

class ConceptExplainerTool(BaseTool):
    name = "explain_concept"
    description = (
        "Get a clear explanation of an AI/ML concept relevant to agentic systems. "
        "Useful for understanding ReAct, chain-of-thought, tool calling, self-reflection, "
        "and agentic workflows."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "concept": {
                "type": "string",
                "description": "The concept to explain, e.g. 'ReAct', 'chain of thought'",
            }
        },
        "required": ["concept"],
    }

    def run(self, concept: str = "") -> ToolResult:
        try:
            key = concept.lower().strip()
            for stored_key, explanation in _CONCEPTS.items():
                if stored_key in key or key in stored_key:
                    return ToolResult(
                        success=True,
                        output={"concept": concept, "explanation": explanation}
                    )
            # Fuzzy fallback
            return ToolResult(
                success=True,
                output={
                    "concept": concept,
                    "explanation": (
                        f"No stored explanation for '{concept}'. "
                        f"Available concepts: {', '.join(_CONCEPTS.keys())}. "
                        "Try rephrasing the concept name."
                    )
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Explainer failed: {e}")


# ─────────────────────────────────────────────
# Tool 4: Word Counter / Text Analyzer
# ─────────────────────────────────────────────

class TextAnalyzerTool(BaseTool):
    name = "analyze_text"
    description = (
        "Analyze a block of text: word count, sentence count, reading level, "
        "keyword frequency, and estimated reading time. "
        "Use this when the task involves analyzing or summarizing text content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze.",
            },
            "top_n_keywords": {
                "type": "integer",
                "description": "How many top keywords to return (default 5).",
            }
        },
        "required": ["text"],
    }

    def run(self, text: str = "", top_n_keywords: int = 5) -> ToolResult:
        try:
            words = re.findall(r"\b[a-zA-Z]+\b", text)
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Keyword frequency (exclude stopwords)
            stopwords = {
                "the","a","an","and","or","but","in","on","at","to","for",
                "of","with","by","from","is","are","was","were","be","been",
                "it","its","this","that","these","those","i","you","we","they",
            }
            freq: dict[str, int] = {}
            for w in words:
                w_lower = w.lower()
                if w_lower not in stopwords and len(w_lower) > 2:
                    freq[w_lower] = freq.get(w_lower, 0) + 1

            top_keywords = sorted(freq.items(), key=lambda x: -x[1])[:top_n_keywords]

            avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
            # Flesch-Kincaid readability approximation
            reading_level = "easy" if avg_word_len < 4.5 else "medium" if avg_word_len < 6 else "advanced"
            reading_time_sec = int(len(words) / 200 * 60)  # 200 WPM average

            return ToolResult(
                success=True,
                output={
                    "word_count": len(words),
                    "sentence_count": len(sentences),
                    "reading_level": reading_level,
                    "estimated_reading_time": f"{reading_time_sec}s",
                    "top_keywords": [{"word": k, "count": v} for k, v in top_keywords],
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Analysis failed: {e}")


# ─────────────────────────────────────────────
# Tool 5: Study Plan Generator
# ─────────────────────────────────────────────

class StudyPlanTool(BaseTool):
    name = "generate_study_plan"
    description = (
        "Generate a structured study plan for a technical topic given "
        "available days and current skill level. Returns a day-by-day plan "
        "with topics, resources, and practice tasks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The technical topic to study.",
            },
            "days_available": {
                "type": "integer",
                "description": "Number of days available for studying.",
            },
            "skill_level": {
                "type": "string",
                "description": "beginner | intermediate | advanced",
            }
        },
        "required": ["topic", "days_available", "skill_level"],
    }

    _PLANS = {
        "claude api": {
            "beginner": [
                {"focus": "API basics", "tasks": ["Install anthropic SDK", "Run first messages.create call", "Understand system vs user messages"]},
                {"focus": "Tool calling", "tasks": ["Define a tool schema", "Parse tool_use response", "Return tool_result", "Build a 2-tool agent"]},
                {"focus": "Agent loop", "tasks": ["Implement ReAct loop", "Add loop safeguards", "Handle tool failures", "Demo walkthrough"]},
            ],
        },
        "agentic workflows": {
            "beginner": [
                {"focus": "Foundations", "tasks": ["What is an agent vs a chatbot", "LLM + tools = agent", "Read ReAct paper abstract"]},
                {"focus": "Build", "tasks": ["Multi-step workflow with 3 tools", "Chain tool outputs", "Self-reflection loop"]},
                {"focus": "Polish", "tasks": ["Add error handling", "Loop detection", "Interview prep Q&A"]},
            ],
        },
    }

    def run(self, topic: str = "", days_available: int = 3, skill_level: str = "beginner") -> ToolResult:
        try:
            topic_key = topic.lower().strip()
            plan_data = None
            for key in self._PLANS:
                if key in topic_key or topic_key in key:
                    plan_data = self._PLANS[key].get(skill_level.lower())
                    break

            if not plan_data:
                # Generic fallback plan
                plan_data = [
                    {"focus": f"Day {i+1}: {topic} fundamentals" if i == 0 else f"Day {i+1}: Practice",
                     "tasks": ["Research core concepts", "Write code", "Review and iterate"]}
                    for i in range(min(days_available, 5))
                ]

            # Trim or expand to match days_available
            days = []
            for i in range(days_available):
                day_info = plan_data[i] if i < len(plan_data) else plan_data[-1]
                days.append({
                    "day": i + 1,
                    "focus": day_info["focus"],
                    "tasks": day_info["tasks"],
                })

            return ToolResult(
                success=True,
                output={
                    "topic": topic,
                    "skill_level": skill_level,
                    "days": days,
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Plan generation failed: {e}")
