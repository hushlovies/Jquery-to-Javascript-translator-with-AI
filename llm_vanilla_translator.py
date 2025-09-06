#!/usr/bin/env python3
"""
Pure LLM jQuery to JavaScript Translator
Uses only GPT-2 Medium for translation without RAG, patterns, or Semgrep
llm_vanilla_translator.py
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")

class LLMVanillaTranslator:
    """Pure LLM-based jQuery to JavaScript translator using GPT-2 Medium"""
    
    def __init__(self, model_name: str = "gpt2-medium"):
        """Initialize translator with GPT-2 model only"""
        self.model_name = model_name
        
        print(f"Loading Pure LLM jQuery to JS Translator")
        print(f"   Model: {model_name}")
        print(f"   Mode: Pure Generative AI (No RAG/Patterns/Security)")
        
        # Load LLM model
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Fix padding token issue - add a new pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
                
            print(f" {self.model_name} loaded on {self.device}")
            
        except Exception as e:
            print(f" Error loading model: {e}")
            raise
    
    def create_translation_prompt(self, jquery_code: str) -> str:
        """Create optimized prompt for LLM translation"""
        prompt = f"""
Convert jQuery to vanilla JavaScript:

Examples:
jQuery: $('#button').click(function() {{ alert('hello'); }});
JavaScript: document.getElementById('button').addEventListener('click', function() {{ alert('hello'); }});

jQuery: $('.items').hide();
JavaScript: document.querySelectorAll('.items').forEach(el => el.style.display = 'none');

jQuery: $('#element').addClass('active');
JavaScript: document.getElementById('element').classList.add('active');

jQuery: $(document).ready(function() {{ init(); }});
JavaScript: document.addEventListener('DOMContentLoaded', function() {{ init(); }});

Convert this jQuery code to vanilla JavaScript:
jQuery: {jquery_code}
JavaScript:"""
        
        return prompt
    
    def generate_with_llm(self, prompt: str, max_length: int = 150) -> str:
        """Generate vanilla JavaScript using GPT-2"""
        try:
            # Encode the prompt with proper attention mask
            encoded = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            inputs = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Generate response with proper parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            generated_part = generated_text[len(prompt):].strip()
            
            return generated_part
            
        except Exception as e:
            print(f"⚠️ Generation error: {str(e)}")
            return f"// Error generating translation: {str(e)}"
    
    def clean_generated_code(self, generated_code: str) -> str:
        """Clean and format the generated JavaScript code"""
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\n\s*\n', '\n', generated_code.strip())
        
        # Remove any incomplete sentences or trailing text after semicolon
        lines = cleaned.split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Stop at first line that doesn't look like code
                if (line.startswith('//') or 
                    'document.' in line or 
                    line.endswith(';') or 
                    line.endswith('{') or 
                    line.endswith('}') or
                    'function' in line or
                    'addEventListener' in line):
                    code_lines.append(line)
                else:
                    # Stop processing if we hit non-code text
                    break
        
        # Join the cleaned lines
        result = '\n'.join(code_lines)
        
        # Ensure proper semicolon ending for single statements
        if result and not result.rstrip().endswith((';', '}', '{')):
            result = result.rstrip() + ';'
        
        return result if result else "// Unable to generate clean translation"
    
    def translate(self, jquery_code: str) -> Dict:
        """Main translation method using pure LLM approach"""
        jquery_code = jquery_code.strip()
        
        if not jquery_code:
            return {
                "original": jquery_code,
                "translated": "// Empty input",
                "method": "none",
                "model": self.model_name
            }
        
        print(f"\n🔄 LLM Translating: {jquery_code}")
        
        try:
            # Create translation prompt
            prompt = self.create_translation_prompt(jquery_code)
            
            # Generate translation using LLM
            generated_code = self.generate_with_llm(prompt)
            
            # Clean the generated code
            cleaned_code = self.clean_generated_code(generated_code)
            
            print(f"LLM translation successful")
            print(f"   Result: {cleaned_code}")
            
            return {
                "original": jquery_code,
                "translated": cleaned_code,
                "method": "llm_generation",
                "model": self.model_name,
                "raw_generation": generated_code  # Keep raw for debugging
            }
            
        except Exception as e:
            print(f" LLM translation failed: {str(e)}")
            return {
                "original": jquery_code,
                "translated": f"// LLM Error: {str(e)}",
                "method": "error",
                "model": self.model_name,
                "error": str(e)
            }
    
    def batch_translate(self, jquery_codes: List[str]) -> List[Dict]:
        """Batch translation using pure LLM approach"""
        results = []
        
        print(f"\n LLM Batch Translation ({len(jquery_codes)} items)")
        print("=" * 60)
        
        for i, code in enumerate(jquery_codes, 1):
            print(f"\n[{i}/{len(jquery_codes)}]", end=" ")
            result = self.translate(code)
            results.append(result)
        
        return results
    
    def translation_report(self, results: List[Dict]):
        """Generate translation report from LLM results"""
        print(f"\n📊 LLM TRANSLATION REPORT")
        print("=" * 40)
        
        # Count by success/failure
        successful = sum(1 for r in results if r['method'] == 'llm_generation')
        failed = sum(1 for r in results if r['method'] == 'error')
        empty = sum(1 for r in results if r['method'] == 'none')
        
        print(f" Successful: {successful}")
        print(f" Failed: {failed}")
        print(f" Empty: {empty}")
        print(f" Success Rate: {successful/len(results)*100:.1f}%")
        print(f" Model Used: {self.model_name}")
    
    def compare_translations(self, jquery_code: str, num_variations: int = 3) -> List[Dict]:
        """Generate multiple translations for comparison (demonstrating AI creativity)"""
        print(f"\n Generating {num_variations} creative variations for: {jquery_code}")
        
        variations = []
        for i in range(num_variations):
            print(f"   Variation {i+1}/{num_variations}...")
            result = self.translate(jquery_code)
            result['variation_id'] = i + 1
            variations.append(result)
        
        return variations
    
    def analyze_creativity(self, variations: List[Dict]) -> Dict:
        """Analyze creativity and diversity in LLM translations"""
        if not variations:
            return {"diversity_score": 0, "unique_approaches": 0}
        
        # Get unique translations
        translations = [v['translated'] for v in variations if v['method'] == 'llm_generation']
        unique_translations = list(set(translations))
        
        diversity_score = len(unique_translations) / len(translations) * 100 if translations else 0
        
        print(f"\n CREATIVITY ANALYSIS")
        print("=" * 30)
        print(f" Total Variations: {len(variations)}")
        print(f" Unique Solutions: {len(unique_translations)}")
        print(f"Diversity Score: {diversity_score:.1f}%")
        
        return {
            "total_variations": len(variations),
            "unique_solutions": len(unique_translations),
            "diversity_score": diversity_score,
            "translations": unique_translations
        }

def main():
    """Test the pure LLM translator"""
    # Initialize translator
    translator = LLMVanillaTranslator(model_name="gpt2-medium")
    
    # Test cases
    test_cases = [
        # Basic cases
        "$('#button').click(function() { console.log('clicked'); });",
        "$('.items').hide();",
        "$('#element').addClass('active');", 
        "$('#text').text('Hello World');",
        "$('#element').css('color', 'red');",
        "$(document).ready(function() { init(); });",
        
        # More complex cases
        "$('.items').each(function(i, el) { $(el).addClass('item-' + i); });",
        "$('#form').submit(function(e) { e.preventDefault(); });",
    ]
    
    print(f"\n Testing Pure LLM jQuery to JavaScript Translator")
    print("=" * 60)
    
    # Translate all test cases
    results = translator.batch_translate(test_cases)
    
    # Generate translation report
    translator.translation_report(results)
    
    # Detailed results
    print(f"\n DETAILED RESULTS")
    print("=" * 40)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['original']}")
        print(f"   → {result['translated']}")
        print(f"   Method: {result['method']}")
        
        if result['method'] == 'error':
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    # Demonstrate creativity analysis
    print(f"\n CREATIVITY DEMONSTRATION")
    print("=" * 40)
    
    # Test creativity on a complex example
    creative_test = "$('#items').each(function() { $(this).fadeIn(); });"
    variations = translator.compare_translations(creative_test, num_variations=3)
    creativity_analysis = translator.analyze_creativity(variations)
    
    print(f"\nVariations generated:")
    for i, var in enumerate(variations, 1):
        print(f"{i}. {var['translated']}")

if __name__ == "__main__":
    main()