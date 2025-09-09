#!/usr/bin/env python3
"""
Pure LLM jQuery→JS Translator (Qwen2.5-Coder-1.5B-Instruct)
CPU-optimized build: shorter context, fewer tokens, KV cache ON, chat-template fallback.
"""

import os
# Force CPU (avoid MPS/Metal crashes)
os.environ["PYTORCH_MPS_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")


# Fallback ChatML template for Qwen-style chat models
QWEN_CHATML_TEMPLATE = """{% if system_message %}<|im_start|>system
{{ system_message }}
<|im_end|>
{% endif %}{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}
<|im_end|>
{% endfor %}<|im_start|>assistant
"""


def _few_shot_prefix() -> str:
    return (
        "Convert jQuery to vanilla JavaScript. Output ONLY the JavaScript code, "
        "no explanations, no labels like 'JavaScript:', and no backticks.\n\n"
        "Examples:\n"
        "jQuery: $('#button').click(function(){ alert('hi'); });\n"
        "JavaScript: document.getElementById('button').addEventListener('click', function(){ alert('hi'); });\n\n"
        "jQuery: $(document).ready(function(){ init(); });\n"
        "JavaScript: document.addEventListener('DOMContentLoaded', function(){ init(); });\n\n"
    )


class LLMVanillaTranslator:
    """Pure LLM-based translator using Qwen2.5-Coder-1.5B-Instruct (CPU-optimized)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        self.model_name = model_name
        print("Loading Pure LLM jQuery→JS Translator")
        print(f"   Model: {model_name}")
        print("   Mode: Pure Generative AI (No RAG/Patterns/Security)")

        # CPU device & threads
        self.device = "cpu"
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass

        # Load tokenizer/model on CPU (float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # Ensure pad token exists (Qwen often uses EOS as pad)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tighter limits for speed on CPU
        self.max_input_tokens = 384   # was 512
        self.max_new_tokens  = 40     # was 96 (one-liners don't need more)

        # KV cache ON -> big speedup on CPU
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

        print(f" {model_name} loaded on CPU")

    # ---------- Prompting ----------

    def create_translation_prompt(self, jquery_code: str) -> str:
        return (_few_shot_prefix() + f"jQuery: {jquery_code}\nJavaScript:").strip()

    def _build_chat(self, prompt: str) -> str:
        system = "You are a concise coding assistant. Output ONLY JavaScript; no prose, no backticks, no labels."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Try official templating; provide fallback if absent
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=getattr(self.tokenizer, "chat_template", None) or QWEN_CHATML_TEMPLATE,
                )
            except Exception:
                pass
        # Manual ChatML
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}\n<|im_end|>\n" for m in messages
        ) + "<|im_start|>assistant\n"

    # ---------- Generation ----------

    def generate_with_llm(self, prompt: str) -> str:
        try:
            chat_prompt = self._build_chat(prompt)
            enc = self.tokenizer(
                chat_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(self.device)

            print(f"   [tokens in] {input_ids.shape[1]}  |  [max new] {self.max_new_tokens}")

            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,      # deterministic
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.05,
                    use_cache=True,       # speed
                )

            gen_ids = out[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return text or "// Unable to generate translation"

        except Exception as e:
            print(f"⚠ Generation error: {e}")
            return f"// Error generating translation: {e}"

    # ---------- Cleaning ----------

    def clean_generated_code(self, generated_code: str) -> str:
        text = generated_code.strip()

        m = re.search(r"```(?:js|javascript)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()

        lines = [ln.strip() for ln in text.splitlines()]
        filtered = []
        for ln in lines:
            if ln.lower() in {"javascript:", "answer:", "output:", "solution:", "code:"}:
                continue
            if ln.startswith("```"):
                continue
            filtered.append(ln)
        text = "\n".join(filtered).strip()

        code_lines = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if (
                s.startswith("//")
                or s.endswith((";", "{", "}"))
                or any(k in s for k in [
                    "document.", "getElementById(", "querySelector(", "querySelectorAll(",
                    "addEventListener(", "function(", "const ", "let ", "var "
                ])
            ):
                code_lines.append(s)

        if code_lines:
            result = "\n".join(code_lines)
        else:
            m2 = re.search(r"(document\.[^\n;]+;)", text)
            result = m2.group(1) if m2 else ""

        result = result.replace("console log", "console.log").strip()
        if result and not result.rstrip().endswith((";", "}", "{")):
            result = result.rstrip() + ";"

        return result if result else "// Unable to generate clean translation"

    # ---------- Public API ----------

    def translate(self, jquery_code: str) -> Dict:
        jquery_code = jquery_code.strip()
        if not jquery_code:
            return {"original": jquery_code, "translated": "// Empty input", "method": "none", "model": self.model_name}

        print(f"\n🔄 LLM Translating: {jquery_code}")
        prompt = self.create_translation_prompt(jquery_code)
        raw = self.generate_with_llm(prompt)
        cleaned = self.clean_generated_code(raw)
        print("LLM translation successful")
        print(f"   Result: {cleaned}")
        return {
            "original": jquery_code,
            "translated": cleaned,
            "method": "llm_generation",
            "model": self.model_name,
            "raw_generation": raw,
        }

    def batch_translate(self, jquery_codes: List[str]) -> List[Dict]:
        print(f"\n LLM Batch Translation ({len(jquery_codes)} items)")
        print("=" * 60)
        return [self.translate(code) for code in jquery_codes]

    def translation_report(self, results: List[Dict]):
        print("\n📊 LLM TRANSLATION REPORT")
        print("=" * 40)
        successful = sum(1 for r in results if r["method"] == "llm_generation")
        failed = sum(1 for r in results if r["method"] == "error")
        empty = sum(1 for r in results if r["method"] == "none")
        total = max(1, len(results))
        print(f" Successful: {successful}")
        print(f" Failed: {failed}")
        print(f" Empty: {empty}")
        print(f" Success Rate: {successful/total*100:.1f}%")
        print(f" Model Used: {self.model_name}")

    def compare_translations(self, jquery_code: str, num_variations: int = 3) -> List[Dict]:
        print(f"\n Generating {num_variations} creative variations for: {jquery_code}")
        return [dict(self.translate(jquery_code), variation_id=i + 1) for i in range(num_variations)]

    def analyze_creativity(self, variations: List[Dict]) -> Dict:
        if not variations:
            return {"diversity_score": 0, "unique_approaches": 0}
        translations = [v["translated"] for v in variations if v["method"] == "llm_generation"]
        uniq = list(set(translations))
        score = (len(uniq) / len(translations) * 100) if translations else 0
        print("\n CREATIVITY ANALYSIS")
        print("=" * 30)
        print(f" Total Variations: {len(variations)}")
        print(f" Unique Solutions: {len(uniq)}")
        print(f" Diversity Score: {score:.1f}%")
        return {"total_variations": len(variations), "unique_solutions": len(uniq), "diversity_score": score, "translations": uniq}


def main():
    t = LLMVanillaTranslator("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    cases = [
        "$('#button').click(function() { console.log('clicked'); });",
        "$('.items').hide();",
        "$('#element').addClass('active');",
        "$('#text').text('Hello World');",
        "$('#element').css('color', 'red');",
        "$(document).ready(function() { init(); });",
        "$('.items').each(function(i, el) { $(el).addClass('item-' + i); });",
        "$('#form').submit(function(e) { e.preventDefault(); });",
    ]
    print("\n Testing Pure LLM jQuery→JS Translator (Qwen2.5-Coder, CPU-optimized)")
    print("=" * 60)
    results = t.batch_translate(cases)
    t.translation_report(results)

    print("\n DETAILED RESULTS")
    print("=" * 40)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['original']}\n   → {r['translated']}\n   Method: {r['method']}")
        if r["method"] == "error":
            print(f"   Error: {r.get('error','Unknown')}")

    print("\n CREATIVITY DEMONSTRATION")
    print("=" * 40)
    creative_test = "$('#items').each(function() { $(this).fadeIn(); });"
    variations = t.compare_translations(creative_test, num_variations=3)
    _ = t.analyze_creativity(variations)
    print("\nVariations generated:")
    for i, v in enumerate(variations, 1):
        print(f"{i}. {v['translated']}")


if __name__ == "__main__":
    main()
