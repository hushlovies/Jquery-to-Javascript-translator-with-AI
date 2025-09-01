#!/usr/bin/env python3
"""
jQuery to JavaScript Translator with Integrated Semgrep Security Validation
Main translator that imports and uses the Semgrep validator module
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import warnings

# Import our Semgrep validator
try:
    from semgrep_validator import SemgrepValidator, SecurityLevel
    SEMGREP_AVAILABLE = True
    print("✅ Semgrep validator imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import Semgrep validator: {e}")
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

def css_to_camel_case(css_prop):
    """Convert CSS property names to camelCase for JavaScript"""
    components = css_prop.split('-')
    return components[0] + ''.join(word.capitalize() for word in components[1:])

class SecurejQueryToJSTranslator:
    """jQuery to JavaScript translator with Semgrep security validation"""
    
    def __init__(self, security_level: str = "MEDIUM", use_semgrep: bool = True):
        """Initialize translator with optional Semgrep security validation"""
        self.model_name = "gpt2-medium"
        self.security_level = security_level  # LOW, MEDIUM, HIGH
        self.use_semgrep = use_semgrep and SEMGREP_AVAILABLE
        
        print(f"🤖 Loading jQuery to JS Translator")
        print(f"   Security Level: {security_level}")
        print(f"   Semgrep Integration: {'✅ Enabled' if self.use_semgrep else '❌ Disabled'}")
        
        # Initialize Semgrep validator if available
        if self.use_semgrep:
            try:
                self.semgrep_validator = SemgrepValidator()
                print("🛡️ Semgrep security validator initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Semgrep: {e}")
                print("   Continuing without Semgrep validation")
                self.use_semgrep = False
        
        # Load LLM model
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
                
            print(f"✅ GPT-2 Medium loaded on {self.device}")
            
            # Load documentation and setup retrieval
            self.load_safe_documentation()
            self.prepare_retrieval_system()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_safe_documentation(self):
        """Load security-verified documentation examples"""
        print("📚 Loading security-verified documentation...")
        
        # Safe documentation examples
        safe_docs = [
            {
                "source": "MDN Web Docs - Safe DOM Manipulation",
                "jquery": "$('#element').addClass('active');",
                "javascript": "document.getElementById('element').classList.add('active');",
                "explanation": "Safe class manipulation using classList"
            },
            {
                "source": "MDN Web Docs - Safe Event Handling",
                "jquery": "$('#button').click(function() { console.log('clicked'); });",
                "javascript": "document.getElementById('button').addEventListener('click', function() { console.log('clicked'); });",
                "explanation": "Safe event handling with addEventListener"
            },
            {
                "source": "MDN Web Docs - Safe Text Content",
                "jquery": "$('#element').text('Safe content');",
                "javascript": "document.getElementById('element').textContent = 'Safe content';",
                "explanation": "Safe text content using textContent (not innerHTML)"
            },
            {
                "source": "MDN Web Docs - Safe Styling",
                "jquery": "$('#element').css('color', 'red');",
                "javascript": "document.getElementById('element').style.color = 'red';",
                "explanation": "Safe direct style manipulation"
            },
            {
                "source": "MDN Web Docs - Safe Element Selection",
                "jquery": "$('.items').hide();",
                "javascript": "document.querySelectorAll('.items').forEach(el => el.style.display = 'none');",
                "explanation": "Safe element selection and manipulation"
            },
            {
                "source": "MDN Web Docs - Safe Document Ready",
                "jquery": "$(document).ready(function() { init(); });",
                "javascript": "document.addEventListener('DOMContentLoaded', function() { init(); });",
                "explanation": "Safe document ready event handling"
            }
        ]
        
        # Validate documentation examples with Semgrep if available
        validated_docs = []
        for doc in safe_docs:
            if self.use_semgrep:
                security_result = self.semgrep_validator.validate_code_security(doc['javascript'])
                if security_result['safe_to_execute']:
                    validated_docs.append(doc)
                else:
                    print(f"⚠️ Excluded unsafe documentation: {doc['source']}")
            else:
                validated_docs.append(doc)  # Add all if no Semgrep validation
        
        self.documentation = validated_docs
        print(f"✅ Loaded {len(self.documentation)} verified examples")
    
    def prepare_retrieval_system(self):
        """Prepare RAG retrieval system"""
        self.docs_text = []
        for doc in self.documentation:
            doc_text = f"jQuery: {doc['jquery']} JavaScript: {doc['javascript']} Explanation: {doc['explanation']}"
            self.docs_text.append(doc_text)
        
        if self.docs_text:
            self.vectorizer = TfidfVectorizer()
            self.doc_vectors = self.vectorizer.fit_transform(self.docs_text)
            print("✅ Retrieval system prepared")
        else:
            print("⚠️ No documentation available for retrieval system")
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """Retrieve relevant documentation with security filtering"""
        if not hasattr(self, 'vectorizer'):
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_docs = []
        for i in top_indices:
            if similarities[i] > 0.1:
                relevant_docs.append(self.documentation[i])
        
        return relevant_docs
    
    def validate_security(self, code: str) -> Tuple[bool, Dict]:
        """Validate JavaScript code security"""
        if self.use_semgrep:
            # Use Semgrep for comprehensive security analysis
            security_result = self.semgrep_validator.validate_code_security(code)
            
            # Apply security level filtering
            if self.security_level == "HIGH":
                is_safe = security_result['level'] == SecurityLevel.SAFE
            elif self.security_level == "MEDIUM":
                is_safe = security_result['level'] in [SecurityLevel.SAFE, SecurityLevel.WARNING]
            else:  # LOW
                is_safe = security_result['level'] != SecurityLevel.DANGEROUS
            
            return is_safe, security_result
        else:
            # Fallback basic validation if Semgrep unavailable
            return self.basic_security_check(code)
    
    def basic_security_check(self, code: str) -> Tuple[bool, Dict]:
        """Basic security check without Semgrep"""
        dangerous_patterns = [
            r'eval\s*\(',
            r'Function\s*\(',
            r'innerHTML\s*=.*\+',
            r'document\.write\s*\(',
            r'__proto__'
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
    
    def secure_pattern_translation(self, jquery_code: str) -> Tuple[str, Dict]:
        """Pattern-based translation with security validation"""
        clean_code = jquery_code.rstrip(';')
        
        # Secure patterns - only safe DOM operations
        secure_patterns = [
            # Safe ID manipulation  
            (r"\$\(['\"]#(.+?)['\"]\)\.addClass\(['\"](.+?)['\"]\)", 
                lambda m: f"document.getElementById('{m.group(1)}').classList.add('{m.group(2)}');"),
            
            (r"\$\(['\"]#(.+?)['\"]\)\.removeClass\(['\"](.+?)['\"]\)", 
                lambda m: f"document.getElementById('{m.group(1)}').classList.remove('{m.group(2)}');"),
            
            # Safe text content (not innerHTML)
            (r"\$\(['\"]#(.+?)['\"]\)\.text\(['\"](.+?)['\"]\)", 
                lambda m: f"document.getElementById('{m.group(1)}').textContent = '{m.group(2)}';"),
            
            # Safe event handling
            (r"\$\(['\"]#(.+?)['\"]\)\.click\(function\(\)\s*\{(.*?)\}\)", 
                lambda m: f"document.getElementById('{m.group(1)}').addEventListener('click', function() {{{m.group(2)}}});"),
            
            # Safe visibility control
            (r"\$\(['\"](.+?)['\"]\)\.hide\(\)", 
                lambda m: f"document.querySelectorAll('{m.group(1)}').forEach(el => el.style.display = 'none');"),
            
            (r"\$\(['\"](.+?)['\"]\)\.show\(\)", 
                lambda m: f"document.querySelectorAll('{m.group(1)}').forEach(el => el.style.display = '');"),
            
            # Safe CSS manipulation
            (r"\$\(['\"]#(.+?)['\"]\)\.css\(['\"](.+?)['\"],\s*['\"](.+?)['\"]\)", 
                lambda m: f"document.getElementById('{m.group(1)}').style.{css_to_camel_case(m.group(2))} = '{m.group(3)}';"),
            
            # Safe class selectors
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
                    is_secure, security_info = self.validate_security(result)
                    if is_secure:
                        return result, security_info
                    else:
                        print(f"⚠️ Pattern result failed security validation")
                except Exception as e:
                    print(f"⚠️ Pattern replacement failed: {e}")
                    continue
        
        return None, {}
    
    def translate(self, jquery_code: str) -> Dict:
        """Main translation method with security validation"""
        jquery_code = jquery_code.strip()
        
        if not jquery_code:
            return {
                "original": jquery_code,
                "translated": "// No code provided",
                "security": {"level": SecurityLevel.SAFE, "safe_to_execute": True, "issues": []},
                "method": "none"
            }
        
        print(f"\n🔄 Translating: {jquery_code}")
        
        # Try secure pattern-based translation
        pattern_result, security_info = self.secure_pattern_translation(jquery_code)
        
        if pattern_result:
            print(f"✅ Secure translation successful")
            print(f"   Result: {pattern_result}")
            
            if self.use_semgrep:
                security_summary = self.semgrep_validator.get_security_summary(pattern_result)
                print(f"   Security: {security_summary}")
            
            return {
                "original": jquery_code,
                "translated": pattern_result,
                "security": security_info,
                "method": "secure_pattern"
            }
        
        # If no secure pattern available, block the translation
        print(f"❌ No secure translation pattern available")
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
        """Batch translation with security validation"""
        results = []
        
        print(f"\n📦 Batch Translation ({len(jquery_codes)} items)")
        print("=" * 60)
        
        for i, code in enumerate(jquery_codes, 1):
            print(f"\n[{i}/{len(jquery_codes)}]", end=" ")
            result = self.translate(code)
            results.append(result)
        
        return results
    
    def security_report(self, results: List[Dict]):
        """Generate security report from translation results"""
        print(f"\n📊 SECURITY REPORT")
        print("=" * 40)
        
        # Count by security level
        safe_count = sum(1 for r in results if r['security']['level'] == SecurityLevel.SAFE)
        warning_count = sum(1 for r in results if r['security']['level'] == SecurityLevel.WARNING)
        dangerous_count = sum(1 for r in results if r['security']['level'] == SecurityLevel.DANGEROUS)
        blocked_count = sum(1 for r in results if r['security']['level'] == SecurityLevel.BLOCKED)
        
        print(f"✅ Safe: {safe_count}")
        print(f"⚠️ Warnings: {warning_count}")
        print(f"❌ Dangerous: {dangerous_count}")
        print(f"⛔ Blocked: {blocked_count}")
        
        # Security score average
        scores = [r['security'].get('score', 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"📈 Average Security Score: {avg_score:.1f}/100")

def main():
    """Test the secure translator"""
    # Initialize translator
    translator = SecurejQueryToJSTranslator(security_level="MEDIUM", use_semgrep=True)
    
    # Test cases
    test_cases = [
        # Safe cases
        "$('#button').click(function() { console.log('safe'); });",
        "$('.items').hide();",
        "$('#element').addClass('active');", 
        "$('#text').text('Safe content');",
        "$('#element').css('color', 'blue');",
        "$(document).ready(function() { init(); });",
        
        # Cases that should be blocked or flagged
        "$('#content').html(userInput);",  # Potentially unsafe
        "$('#element').attr('onclick', 'alert(1)');",  # Event injection
    ]
    
    print(f"\n🧪 Testing Secure jQuery to JavaScript Translator")
    print("=" * 60)
    
    # Translate all test cases
    results = translator.batch_translate(test_cases)
    
    # Generate security report
    translator.security_report(results)
    
    # Detailed results
    print(f"\n📝 DETAILED RESULTS")
    print("=" * 40)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['original']}")
        print(f"   → {result['translated']}")
        security_level = result['security']['level'].value
        print(f"   Security: {security_level}")
        
        # Show security issues if any
        issues = result['security'].get('issues', [])
        if issues:
            for issue in issues[:2]:  # Show first 2 issues
                print(f"   Issue: {issue.get('message', 'Unknown issue')}")

if __name__ == "__main__":
    main()