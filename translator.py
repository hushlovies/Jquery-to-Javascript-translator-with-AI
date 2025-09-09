#!/usr/bin/env python3
"""
jQuery to JavaScript Translator with Integrated Semgrep Security Validation
- Secure, pattern-based translator (regex + JSON-like rules)
- Semgrep validation
- Loads Qwen2.5-Coder-1.5B-Instruct (CPU-only) for future use / parity (LLM is secondary)
translator.py
"""

import os
# Avoid Metal (MPS) crashes on macOS by forcing CPU for Transformers/PyTorch
os.environ["PYTORCH_MPS_DISABLE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import warnings

# ---------------------------------------------------------------------
# Semgrep validator (optional)
# ---------------------------------------------------------------------
try:
    from semgrep_validator import SemgrepValidator, SecurityLevel
    SEMGREP_AVAILABLE = True
    print(" Semgrep validator imported successfully")
except ImportError as e:
    print(f"Could not import Semgrep validator: {e}")
    print("   Make sure semgrep_validator.py is in the same directory")
    SEMGREP_AVAILABLE = False

    # Fallback SecurityLevel enum if import fails
    from enum import Enum
    class SecurityLevel(Enum):
        SAFE = "SAFE"
        WARNING = "WARNING"
        DANGEROUS = "DANGEROUS"
        BLOCKED = "BLOCKED"

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def css_to_camel_case(css_prop: str) -> str:
    """Convert CSS property names to camelCase for JavaScript"""
    components = css_prop.split('-')
    return components[0] + ''.join(word.capitalize() for word in components[1:])


# ---------------------------------------------------------------------
# Main secure translator
# ---------------------------------------------------------------------
class SecurejQueryToJSTranslator:
    """jQuery → JavaScript translator with Semgrep security validation"""

    def __init__(self, security_level: str = "MEDIUM", use_semgrep: bool = True):
        """
        Initialize translator with optional Semgrep security validation.
        Loads Qwen2.5-Coder (chat code model) on CPU for parity with your setup.
        (LLM is not used to generate code here; patterns + Semgrep are the source of truth.)
        """
        # --- Settings ---------------------------------------------------
        self.model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        self.security_level = security_level  # "LOW" | "MEDIUM" | "HIGH"
        self.use_semgrep = use_semgrep and SEMGREP_AVAILABLE

        print(" Loading jQuery → JS Translator (secure patterns + Semgrep)")
        print(f"   Security Level: {security_level}")
        print(f"   Semgrep Integration: {'Enabled' if self.use_semgrep else '❌ Disabled'}")

        # --- Semgrep setup ----------------------------------------------
        if self.use_semgrep:
            try:
                self.semgrep_validator = SemgrepValidator()
                print(" Semgrep security validator initialized")
            except Exception as e:
                print(f" Failed to initialize Semgrep: {e}")
                print("   Continuing without Semgrep validation")
                self.use_semgrep = False

        # --- LLM (Qwen) load (CPU-only, safe defaults) ------------------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,      # CPU-friendly
                low_cpu_mem_usage=True
            )
            self.device = torch.device("cpu")  # stable for Mac
            self.model.to(self.device).eval()

            # Ensure pad token exists for clean generation if you ever use it
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f" Qwen2.5-Coder loaded on {self.device} (CPU)")
        except Exception as e:
            print(f" Error loading Qwen model: {e}")
            print("   Continuing without LLM (patterns + Semgrep still work).")
            self.tokenizer, self.model, self.device = None, None, "cpu"

        # --- Load secure documentation and prepare retrieval ------------
        self.load_safe_documentation()
        self.prepare_retrieval_system()

    # ------------------------------------------------------------------
    # Documentation (safe examples) + Retrieval (TF-IDF)
    # ------------------------------------------------------------------
    def load_safe_documentation(self):
        """Load security-verified documentation examples (MDN/OWASP-minded)."""
        print(" Loading security-verified documentation...")

        safe_docs = [
            {
                "source": "MDN Web Docs - Safe DOM Manipulation (classList.add)",
                "url": "https://developer.mozilla.org/en-US/docs/Web/API/Element/classList",
                "jquery_url": "https://api.jquery.com/addClass/",
                "jquery": "$('#element').addClass('active');",
                "javascript": "document.getElementById('element').classList.add('active');",
                "explanation": "Safe class manipulation using classList"
            },
            {
                "source": "MDN Web Docs - Safe Event Handling (addEventListener)",
                "url": "https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener",
                "jquery_url": "https://api.jquery.com/click/",
                "jquery": "$('#button').click(function() { console.log('clicked'); });",
                "javascript": "document.getElementById('button').addEventListener('click', function() { console.log('clicked'); });",
                "explanation": "Safe event handling with addEventListener"
            },
            {
                "source": "MDN Web Docs - Safe Text Content (textContent, not innerHTML)",
                "urls": [
                    "https://developer.mozilla.org/en-US/docs/Web/API/Node/textContent",
                    "https://developer.mozilla.org/en-US/docs/Web/API/Element/innerHTML"
                ],
                "security_urls": [
                    "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html"
                ],
                "jquery_url": "https://api.jquery.com/text/",
                "jquery": "$('#element').text('Safe content');",
                "javascript": "document.getElementById('element').textContent = 'Safe content';",
                "explanation": "Prefer textContent to avoid HTML execution and XSS risks"
            },
            {
                "source": "MDN Web Docs - Safe Styling (CSSOM inline style)",
                "url": "https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/style",
                "jquery_url": "https://api.jquery.com/css/",
                "jquery": "$('#element').css('color', 'red');",
                "javascript": "document.getElementById('element').style.color = 'red';",
                "explanation": "Direct style manipulation via CSSStyleDeclaration"
            },
            {
                "source": "MDN Web Docs - Safe Element Selection (querySelectorAll + forEach)",
                "urls": [
                    "https://developer.mozilla.org/en-US/docs/Web/API/Document/querySelectorAll",
                    "https://developer.mozilla.org/en-US/docs/Web/API/NodeList/forEach"
                ],
                "jquery_url_hide": "https://api.jquery.com/hide/",
                "jquery_url_show": "https://api.jquery.com/show/",
                "jquery": "$('.items').hide();",
                "javascript": "document.querySelectorAll('.items').forEach(el => el.style.display = 'none');",
                "explanation": "Select elements with CSS selectors and iterate safely"
            },
            {
                "source": "MDN Web Docs - Safe Document Ready (DOMContentLoaded)",
                "url": "https://developer.mozilla.org/en-US/docs/Web/API/Document/DOMContentLoaded_event",
                "jquery_url": "https://api.jquery.com/ready/",
                "jquery": "$(document).ready(function() { init(); });",
                "javascript": "document.addEventListener('DOMContentLoaded', function() { init(); });",
                "explanation": "Run code after DOM is parsed (native equivalent of jQuery ready)"
            }
        ]

        # Validate documentation examples with Semgrep if available
        validated_docs = []
        for doc in safe_docs:
            if self.use_semgrep:
                try:
                    security_result = self.semgrep_validator.validate_code_security(doc["javascript"])
                    if security_result.get("safe_to_execute", False):
                        validated_docs.append(doc)
                    else:
                        print(f" ⚠️ Excluded unsafe documentation: {doc['source']}")
                except Exception as e:
                    print(f" ⚠️ Semgrep validation failed on doc: {doc['source']} ({e})")
                    # In doubt, keep it only if Semgrep failed for technical reasons
                    validated_docs.append(doc)
            else:
                validated_docs.append(doc)

        self.documentation = validated_docs
        print(f" Loaded {len(self.documentation)} verified examples")

    def prepare_retrieval_system(self):
        """Prepare mini retrieval (TF-IDF) over the safe examples."""
        self.docs_text = []
        for doc in self.documentation:
            doc_text = f"jQuery: {doc['jquery']} JavaScript: {doc['javascript']} Explanation: {doc['explanation']}"
            self.docs_text.append(doc_text)

        if self.docs_text:
            self.vectorizer = TfidfVectorizer()
            self.doc_vectors = self.vectorizer.fit_transform(self.docs_text)
            print(" Retrieval system prepared")
        else:
            print(" No documentation available for retrieval system")

    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documentation (cosine similarity on TF-IDF)."""
        if not hasattr(self, "vectorizer"):
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        relevant_docs = []
        for i in top_indices:
            if similarities[i] > 0.1:
                relevant_docs.append(self.documentation[i])

        return relevant_docs

    # ------------------------------------------------------------------
    # Security validation
    # ------------------------------------------------------------------
    def validate_security(self, code: str) -> Tuple[bool, Dict]:
        """Validate JavaScript code security with Semgrep or basic checks."""
        if self.use_semgrep:
            security_result = self.semgrep_validator.validate_code_security(code)

            # Apply security level policy
            if self.security_level == "HIGH":
                is_safe = security_result["level"] == SecurityLevel.SAFE
            elif self.security_level == "MEDIUM":
                is_safe = security_result["level"] in [SecurityLevel.SAFE, SecurityLevel.WARNING]
            else:  # LOW
                is_safe = security_result["level"] != SecurityLevel.DANGEROUS

            return is_safe, security_result

        # Fallback basic validator
        return self.basic_security_check(code)

    def basic_security_check(self, code: str) -> Tuple[bool, Dict]:
        """Basic security check without Semgrep."""
        dangerous_patterns = [
            r"eval\s*\(",
            r"Function\s*\(",
            r"innerHTML\s*=.*\+",
            r"document\.write\s*\(",
            r"__proto__",
        ]

        issues = []
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                issues.append({
                    "type": "BASIC_CHECK",
                    "pattern": pattern,
                    "message": f"Potentially dangerous pattern: {pattern}",
                    "severity": "WARNING"
                })

        is_safe = len(issues) == 0
        return is_safe, {
            "level": SecurityLevel.SAFE if is_safe else SecurityLevel.WARNING,
            "issues": issues,
            "score": 100 - (len(issues) * 20),
            "safe_to_execute": is_safe,
            "tool": "basic_check"
        }

    # ------------------------------------------------------------------
    # Secure, deterministic pattern-based translation
    # ------------------------------------------------------------------
    def secure_pattern_translation(self, jquery_code: str) -> Tuple[str, Dict]:
        """Pattern-based translation with security validation (deterministic)."""
        clean_code = jquery_code.rstrip(";")

        secure_patterns = [
            # ID selectors: add/remove class
            (r"\$\(['\"]#(.+?)['\"]\)\.addClass\(['\"](.+?)['\"]\)",
             lambda m: f"document.getElementById('{m.group(1)}').classList.add('{m.group(2)}');"),

            (r"\$\(['\"]#(.+?)['\"]\)\.removeClass\(['\"](.+?)['\"]\)",
             lambda m: f"document.getElementById('{m.group(1)}').classList.remove('{m.group(2)}');"),

            # Safe text content (prefer textContent)
            (r"\$\(['\"]#(.+?)['\"]\)\.text\(['\"](.+?)['\"]\)",
             lambda m: f"document.getElementById('{m.group(1)}').textContent = '{m.group(2)}';"),

            # Safe event handling (simple click with inline function)
            (r"\$\(['\"]#(.+?)['\"]\)\.click\(function\(\)\s*\{(.*?)\}\)",
             lambda m: f"document.getElementById('{m.group(1)}').addEventListener('click', function() {{{m.group(2)}}});"),

            # Class / tag selectors: hide/show
            (r"\$\(['\"](.+?)['\"]\)\.hide\(\)",
             lambda m: f"document.querySelectorAll('{m.group(1)}').forEach(el => el.style.display = 'none');"),

            (r"\$\(['\"](.+?)['\"]\)\.show\(\)",
             lambda m: f"document.querySelectorAll('{m.group(1)}').forEach(el => el.style.display = '');"),

            # CSS manipulation by ID
            (r"\$\(['\"]#(.+?)['\"]\)\.css\(['\"](.+?)['\"],\s*['\"](.+?)['\"]\)",
             lambda m: f"document.getElementById('{m.group(1)}').style.{css_to_camel_case(m.group(2))} = '{m.group(3)}';"),

            # CSS manipulation by class
            (r"\$\(['\"]\.(.+?)['\"]\)\.css\(['\"](.+?)['\"],\s*['\"](.+?)['\"]\)",
             lambda m: f"document.querySelectorAll('.{m.group(1)}').forEach(el => el.style.{css_to_camel_case(m.group(2))} = '{m.group(3)}');"),

            # Document ready
            (r"\$\(document\)\.ready\(function\(\)\s*\{(.*?)\}\)",
             lambda m: f"document.addEventListener('DOMContentLoaded', function() {{{m.group(1)}}});"),
        ]

        for pattern, replacement in secure_patterns:
            match = re.search(pattern, clean_code)
            if match:
                try:
                    result = replacement(match)
                    # Small auto-fix for a common slip in examples
                    result = result.replace("console log", "console.log")

                    is_secure, security_info = self.validate_security(result)
                    if is_secure:
                        return result, security_info
                    else:
                        print(" ⚠️ Pattern result failed security validation")
                except Exception as e:
                    print(f" ⚠️ Pattern replacement failed: {e}")
                    continue

        return None, {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def translate(self, jquery_code: str) -> Dict:
        """Main translation method with security validation."""
        jquery_code = jquery_code.strip()

        if not jquery_code:
            return {
                "original": jquery_code,
                "translated": "// Not in the pattern",
                "security": {"level": SecurityLevel.SAFE, "safe_to_execute": True, "issues": []},
                "method": "none"
            }

        print(f"\n Translating: {jquery_code}")

        # 1) Try secure pattern-based translation
        pattern_result, security_info = self.secure_pattern_translation(jquery_code)

        if pattern_result:
            print(" Secure translation successful")
            print(f"   Result: {pattern_result}")

            if self.use_semgrep:
                try:
                    security_summary = self.semgrep_validator.get_security_summary(pattern_result)
                    print(f"   Security: {security_summary}")
                except Exception as e:
                    print(f"   (Semgrep summary failed: {e})")

            return {
                "original": jquery_code,
                "translated": pattern_result,
                "security": security_info,
                "method": "secure_pattern"
            }

        # 2) No secure pattern → block (security-first)
        print(" No secure translation pattern available")
        return {
            "original": jquery_code,
            "translated": f"// BLOCKED: No secure translation available for: {jquery_code}",
            "security": {
                "level": SecurityLevel.BLOCKED,
                "safe_to_execute": False,
                "issues": [{"type": "NO_SECURE_PATTERN", "description": "No secure translation pattern available"}],
                "tool": "translator"
            },
            "method": "blocked"
        }

    def batch_translate(self, jquery_codes: List[str]) -> List[Dict]:
        """Batch translation with security validation."""
        results = []

        print(f"\n Batch Translation ({len(jquery_codes)} items)")
        print("=" * 60)

        for i, code in enumerate(jquery_codes, 1):
            print(f"\n[{i}/{len(jquery_codes)}]", end=" ")
            result = self.translate(code)
            results.append(result)

        return results

    def security_report(self, results: List[Dict]):
        """Generate security report from translation results."""
        print("\n SECURITY REPORT")
        print("=" * 40)

        safe_count = sum(1 for r in results if r["security"]["level"] == SecurityLevel.SAFE)
        warning_count = sum(1 for r in results if r["security"]["level"] == SecurityLevel.WARNING)
        dangerous_count = sum(1 for r in results if r["security"]["level"] == SecurityLevel.DANGEROUS)
        blocked_count = sum(1 for r in results if r["security"]["level"] == SecurityLevel.BLOCKED)

        print(f" Safe: {safe_count}")
        print(f" Warnings: {warning_count}")
        print(f" Dangerous: {dangerous_count}")
        print(f" Blocked: {blocked_count}")

        scores = [r["security"].get("score", 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f" Average Security Score: {avg_score:.1f}/100")


# ---------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------
def main():
    translator = SecurejQueryToJSTranslator(security_level="MEDIUM", use_semgrep=True)

    test_cases = [
        # Safe cases
        "$('#button').click(function() { console.log('safe'); });",
        "$('.items').hide();",
        "$('#element').addClass('active');",
        "$('#text').text('Safe content');",
        "$('#element').css('color', 'blue');",
        "$(document).ready(function() { init(); });",

        # Cases that should be blocked or flagged
        "$('#content').html(userInput);",                  # Potentially unsafe
        "$('#element').attr('onclick', 'alert(1)');",      # Event injection
    ]

    print("\n Testing Secure jQuery → JavaScript Translator")
    print("=" * 60)

    results = translator.batch_translate(test_cases)
    translator.security_report(results)

    print("\n DETAILED RESULTS")
    print("=" * 40)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['original']}")
        print(f"   → {result['translated']}")
        level = result["security"]["level"].value if hasattr(result["security"]["level"], "value") else str(result["security"]["level"])
        print(f"   Security: {level}")
        issues = result["security"].get("issues", [])
        if issues:
            for issue in issues[:2]:
                print(f"   Issue: {issue.get('message', issue)}")

if __name__ == "__main__":
    main()
