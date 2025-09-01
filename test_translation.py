import re

def test_translation():
    """Test the regex pattern matching"""
    
    transformation_rules = {
        # ID Selectors
        r"\$\s*\(\s*['\"]#([^'\"]+)['\"]\s*\)": r"document.getElementById('\1')",
        
        # CSS Classes
        r"\.addClass\s*\(\s*['\"]([^'\"]+)['\"]\s*\)": r".classList.add('\1')",
        r"\.removeClass\s*\(\s*['\"]([^'\"]+)['\"]\s*\)": r".classList.remove('\1')",
        
        # Events
        r"\.click\s*\(\s*function\s*\(\s*([^)]*)\s*\)\s*\{": r".addEventListener('click', function(\1) {",
        
        # DOM manipulation
        r"\.hide\s*\(\s*\)": r".style.display = 'none'",
        r"\.val\s*\(\s*\)": r".value",
    }
    
    test_cases = [
        "$('#toggle').addClass('active');",
        "$('#button').click(function() { alert('Hi'); });",
        "$('.warning').hide();",
        "$('#input').val();",
        "$('#element').removeClass('old');"
    ]
    
    print("🧪 Testing jQuery → Vanilla JS Translation")
    print("=" * 60)
    
    for i, jquery_code in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {jquery_code}")
        
        translated = jquery_code
        applied_rules = []
        
        # Apply transformation rules
        for pattern, replacement in transformation_rules.items():
            if re.search(pattern, translated):
                old_code = translated
                translated = re.sub(pattern, replacement, translated)
                
                if old_code != translated:
                    applied_rules.append(pattern)
                    print(f"✅ Applied rule: {pattern}")
        
        print(f"🎯 Result: {translated}")
        
        if translated != jquery_code:
            print("✅ SUCCESS - Translation applied!")
        else:
            print("❌ FAILED - No translation occurred")
        
        print("-" * 60)
    
    print("\n🎉 Translation test complete!")

def test_specific_case():
    """Test the specific case user mentioned"""
    
    jquery_code = "$('#toggle').addClass('active');"
    expected = "document.getElementById('toggle').classList.add('active');"
    
    print(f"🎯 Testing specific case:")
    print(f"Input:    {jquery_code}")
    print(f"Expected: {expected}")
    
    # Apply transformations step by step
    result = jquery_code
    
    # Step 1: Replace $('#toggle') with document.getElementById('toggle')
    id_pattern = r"\$\s*\(\s*['\"]#([^'\"]+)['\"]\s*\)"
    id_replacement = r"document.getElementById('\1')"
    
    if re.search(id_pattern, result):
        result = re.sub(id_pattern, id_replacement, result)
        print(f"Step 1:   {result}")
    
    # Step 2: Replace .addClass('active') with .classList.add('active')
    class_pattern = r"\.addClass\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
    class_replacement = r".classList.add('\1')"
    
    if re.search(class_pattern, result):
        result = re.sub(class_pattern, class_replacement, result)
        print(f"Step 2:   {result}")
    
    print(f"Final:    {result}")
    
    if result == expected:
        print("✅ SUCCESS - Perfect match!")
    else:
        print("❌ FAILED - Doesn't match expected")
    
    return result == expected

if __name__ == "__main__":
    print("🔧 Quick Translation Test")
    print("=" * 40)
    
    # Test the specific case first
    success = test_specific_case()
    
    print("\n" + "=" * 40)
    
    # Test multiple cases
    test_translation()
    
    if success:
        print("\n🎉 The fix should work! Update your files and try again.")
    else:
        print("\n⚠️ Need to debug the regex patterns more.")