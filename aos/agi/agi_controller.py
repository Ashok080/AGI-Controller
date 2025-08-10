Save as aos/agi/agi_controller.py

"""
agi_controller.py
AGI 2.0 core controller for AOS:
- Lightweight wrapper for LLMs (OpenAI by default)
- Memory interface (short-term + vector long-term)
- Tool-calling helper
- Multimodal placeholders (image/audio)
- Self-reflection / critic step

Usage:
    from aos.agi.agi_controller import AGIController
    agi = AGIController(openai_api_key="sk-...")
    out = agi.ask("Explain how a rocket works.")
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional

import openai

# Optional imports (used only if available)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Basic logger
logger = logging.getLogger("agi_controller")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


class AGIController:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",  # change to available model
        long_term_store: Optional[Any] = None,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        openai_api_key: your OpenAI key or set OPENAI_API_KEY env var
        model: model name to call (switch per provider)
        long_term_store: optional object with .search(query_embedding) and .add(items)
        embed_model_name: model used for embeddings if you use SentenceTransformer
        """
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY or pass openai_api_key.")
        openai.api_key = openai_api_key

        self.model = model
        self.short_memory: List[Dict[str, str]] = []  # simple in-memory chat history
        self.long_term_store = long_term_store
        self.embed_model = None
        if SentenceTransformer is not None:
            try:
                self.embed_model = SentenceTransformer(embed_model_name)
            except Exception as e:
                logger.warning("SentenceTransformer init failed: %s", e)
                self.embed_model = None

        logger.info("AGIController initialized with model=%s", self.model)

    # ---------------------------
    # Memory helpers
    # ---------------------------
    def append_short_memory(self, role: str, content: str):
        self.short_memory.append({"role": role, "content": content})
        # keep short memory to last N messages
        if len(self.short_memory) > 40:
            self.short_memory = self.short_memory[-40:]

    def clear_short_memory(self):
        self.short_memory = []

    def embed_text(self, text: str):
        if self.embed_model:
            return self.embed_model.encode(text).tolist()
        # fallback: return None
        return None

    def retrieve_long_term(self, query: str, top_k: int = 5):
        """
        If long_term_store is provided, use it to fetch relevant memory.
        long_term_store must implement `.search(embedding, top_k)` or `.search_text(query, top_k)`
        """
        if not self.long_term_store:
            return []
        emb = self.embed_text(query)
        if emb is not None and hasattr(self.long_term_store, "search"):
            return self.long_term_store.search(emb, top_k=top_k)
        if hasattr(self.long_term_store, "search_text"):
            return self.long_term_store.search_text(query, top_k=top_k)
        return []

    def add_to_long_term(self, text: str, metadata: Optional[Dict] = None):
        if not self.long_term_store:
            return
        emb = self.embed_text(text)
        item = {"text": text, "embedding": emb, "meta": metadata or {}}
        if hasattr(self.long_term_store, "add"):
            self.long_term_store.add(item)
        else:
            logger.warning("long_term_store has no add() method")

    # ---------------------------
    # Tool call helper
    # ---------------------------
    def call_tool(self, tool_fn, *args, **kwargs):
        """
        Generic tool caller: tool_fn should be a function that returns a dict or string.
        Useful for web_search, db_query, calculator, etc.
        """
        try:
            logger.info("Calling tool: %s", getattr(tool_fn, "__name__", str(tool_fn)))
            return tool_fn(*args, **kwargs)
        except Exception as e:
            logger.exception("Tool call failed: %s", e)
            return {"error": str(e)}

    # ---------------------------
    # Multimodal placeholders
    # ---------------------------
    def analyze_image(self, image_bytes: bytes) -> str:
        """
        Placeholder: run image -> caption/ocr/vision model.
        Implement using OpenAI vision or local vision model.
        """
        # Example: return a placeholder caption
        return "Image analysis placeholder — implement with vision model (BLIP/CLIP/OpenAI Vision)."

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Placeholder: transcribe audio using Whisper or other STT.
        """
        # If whisper package installed you might call it here
        return "Audio transcription placeholder — implement with whisper or cloud STT."

    # ---------------------------
    # LLM interaction
    # ---------------------------
    def _build_system_message(self, system_prompt: Optional[str]):
        if system_prompt:
            return {"role": "system", "content": system_prompt}
        return {"role": "system", "content": "You are a helpful, honest AI assistant."}

    def ask(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_memory: bool = True,
        tools: Optional[List] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Main entrypoint: assemble context, call LLM, optionally perform critic/self-reflection.
        Returns dict with keys: answer, raw, metadata
        """
        messages = []
        messages.append(self._build_system_message(system_prompt))
        # add short-term memory
        if use_memory and self.short_memory:
            # include last few messages as context
            messages.extend(self.short_memory[-10:])

        # optionally retrieve long-term memory and add as context
        if use_memory and self.long_term_store:
            mems = self.retrieve_long_term(prompt, top_k=3)
            if mems:
                messages.append({"role": "system", "content": f"Relevant memory: {json.dumps(mems)}"})

        messages.append({"role": "user", "content": prompt})

        logger.info("Sending prompt to LLM (len messages=%d) ...", len(messages))
        start = time.time()
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.time() - start
            raw_text = response.choices[0].message.content.strip()
            logger.info("LLM returned in %.2fs", elapsed)
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return {"answer": "", "error": str(e), "raw": None}

        # store in short memory
        self.append_short_memory("user", prompt)
        self.append_short_memory("assistant", raw_text)

        result = {"answer": raw_text, "raw": response}

        # optional self-reflection / critic
        crit = self.self_reflect(prompt, raw_text, system_prompt=system_prompt)
        result["self_reflection"] = crit

        return result

    # ---------------------------
    # Self-reflection / Critic
    # ---------------------------
    def self_reflect(self, prompt: str, answer: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask the model to critique its own answer and optionally improve it.
        Returns a dict with critique and optionally an improved_answer.
        """
        critique_prompt = (
            "You are a critic. Evaluate the assistant's answer for correctness, completeness, and possible hallucinations. "
            "Respond in JSON with keys: score (0-10), critique, suggestions, should_improve (true/false)."
            f"\n\nUser Prompt:\n{prompt}\n\nAssistant Answer:\n{answer}\n"
        )
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[self._build_system_message(system_prompt), {"role": "user", "content": critique_prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            critique_text = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("Critic call failed: %s", e)
            return {"error": str(e)}

        # naive parse: try to parse JSON if the model returned JSON
        parsed = {"raw": critique_text}
        try:
            # attempt to find first JSON object in the text
            import re
            m = re.search(r"\{.*\}", critique_text, re.DOTALL)
            if m:
                parsed_json = json.loads(m.group(0))
                parsed.update(parsed_json)
        except Exception:
            # keep raw if parse fails
            parsed["parse_error"] = "Could not parse JSON from critic"
        return parsed


# --- Example tool: simple web search (placeholder) ---
def simple_web_search(query: str) -> Dict[str, Any]:
    """
    Placeholder search implementation. Replace with SERPAPI or DuckDuckGo or Bing.
    """
    return {"query": query, "results": ["result1 snippet...", "result2 snippet..."]}


# Quick test when run as script
if __name__ == "__main__":
    key = os.getenv("OPENAI_API_KEY", None)
    if not key:
        print("Set OPENAI_API_KEY env var first.")
        raise SystemExit(1)
    agen = AGIController(openai_api_key=key)
    out = agen.ask("Explain in simple terms how photosynthesis works.")
    print("Answer:", out.get("answer"))
    print("Self reflection:", out.get("self_reflection"))
