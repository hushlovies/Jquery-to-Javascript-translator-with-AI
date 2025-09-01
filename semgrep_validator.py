#!/usr/bin/env python3
"""
Enhanced translator.py with detailed Semgrep security output
Shows exactly why code is blocked with full explanations
"""

# Add this enhanced method to your SecurejQueryToJSTranslator class

def display_detailed_security_info(self, security_result: dict, code: str):
    """Display detailed security information from Semgrep"""
    
    print(f"\n🛡️ SECURITY ANALYSIS DETAILS")
    print("=" * 50)
    print(f"Code: {code}")
    print(f"Overall Security Level: {security_result['level'].value}")
    print(f"Security Score: {security_result['score']}/100")
    print(f"Safe to Execute: {'Yes' if security_result['safe_to_execute'] else 'No'}")
    print(f"Analysis Tool: {security_result.get('tool', 'unknown')}")
    
    # Show detailed issues
    issues = security_result.get('issues', [])
    if issues:
        print(f"\n📋 SECURITY ISSUES FOUND ({len(issues)}):")
        print("-" * 40)
        
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. ISSUE: {issue.get('type', 'Unknown')}")
            print(f"   Rule ID: {issue.get('rule_id', 'N/A')}")
            print(f"   Severity: {issue.get('severity', 'N/A')}")
            print(f"   Message: {issue.get('message', 'No description available')}")
            
            if issue.get('line'):
                print(f"   Location: Line {issue['line']}, Column {issue.get('column', 0)}")
            
            # Show code context if available
            if issue.get('code'):
                print(f"   Code Context: {issue['code']}")
    else:
        print("\n✅ No security issues detected")
    
    # Show recommendations
    recommendations = security_result.get('recommendations', [])
    if recommendations:
        print(f"\n💡 SECURITY RECOMMENDATIONS:")
        print("-" * 30)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

def enhanced_translate(self, jquery_code: str) -> dict:
    """Enhanced translation with detailed security reporting"""
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
        print(f"✅ Pattern-based translation successful")
        print(f"   Result: {pattern_result}")
        
        # Show detailed security analysis
        if self.use_semgrep and security_info:
            self.display_detailed_security_info(security_info, pattern_result)
        
        return {
            "original": jquery_code,
            "translated": pattern_result,
            "security": security_info,
            "method": "secure_pattern"
        }
    
    # If no pattern found, run security analysis on original jQuery code
    # to show WHY it's blocked
    print(f"❌ No secure translation pattern available")
    
    # Analyze the jQuery code to understand why it's blocked
    if self.use_semgrep:
        # Convert jQuery to basic JS first to analyze security
        basic_js = self.jquery_to_basic_js(jquery_code)
        security_analysis = self.semgrep_validator.validate_code_security(basic_js)
        
        print(f"\n🔍 SECURITY ANALYSIS OF ORIGINAL CODE:")
        self.display_detailed_security_info(security_analysis, basic_js)
    
    return {
        "original": jquery_code,
        "translated": f"// BLOCKED: No secure translation available for: {jquery_code}",
        "security": {
            "level": SecurityLevel.BLOCKED,
            "safe_to_execute": False,
            "issues": [{"type": "NO_SECURE_PATTERN", "description": "No secure translation pattern available - see analysis above"}],
            "tool": "translator"
        },
        "method": "blocked"
    }

def jquery_to_basic_js(self, jquery_code: str) -> str:
    """Convert jQuery to basic JavaScript for security analysis (unsafe conversion)"""
    # This is just for security analysis - NOT for production use
    conversions = {
        r"\$\(['\"]#(.+?)['\"]\)\.html\((.+?)\)": r"document.getElementById('\1').innerHTML = \2",
        r"\$\(['\"]#(.+?)['\"]\)\.text\((.+?)\)": r"document.getElementById('\1').textContent = \2",
        r"\$\(['\"]#(.+?)['\"]\)\.click\((.+?)\)": r"document.getElementById('\1').addEventlistener('click', \2)",
        r"\$\(['\"]\.(.+?)['\"]\)\.html\((.+?)\)": r"document.querySelectorAll('.\1').forEach(el => el.innerHTML = \2)",
    }
    
    result = jquery_code
    for pattern, replacement in conversions.items():
        result = re.sub(pattern, replacement, result)
    
    return result

# Enhanced security validator that provides more details
def enhanced_validate_code_security(self, code: str) -> dict:
    """Enhanced version that captures more details from Semgrep"""
    
    if not code or not code.strip():
        return {
            "level": SecurityLevel.SAFE,
            "issues": [],
            "score": 100,
            "safe_to_execute": True,
            "tool": "semgrep"
        }
    
    # Create temporary file
    js_file = self.create_temp_js_file(code)
    
    try:
        # Run Semgrep analysis with verbose output
        semgrep_result = self.run_enhanced_semgrep_analysis(js_file)
        
        # Process results with enhanced detail extraction
        issues = []
        security_level = SecurityLevel.SAFE
        
        for finding in semgrep_result.get("results", []):
            extra = finding.get("extra", {})
            severity = extra.get("severity", "INFO")
            
            # Extract more detailed information
            issue = {
                "type": "SEMGREP_FINDING",
                "rule_id": finding.get("check_id", "unknown"),
                "message": extra.get("message", "Security issue detected"),
                "severity": severity,
                "line": finding.get("start", {}).get("line", 0),
                "column": finding.get("start", {}).get("col", 0),
                "end_line": finding.get("end", {}).get("line", 0),
                "code": finding.get("extra", {}).get("lines", ""),
                "fix": extra.get("fix", ""),
                "metadata": extra.get("metadata", {}),
                "confidence": extra.get("confidence", "MEDIUM"),
                "category": extra.get("category", "security"),
                "cwe": extra.get("metadata", {}).get("cwe", []),
                "owasp": extra.get("metadata", {}).get("owasp", []),
                "references": extra.get("metadata", {}).get("references", [])
            }
            
            issues.append(issue)
            
            # Update security level based on severity
            if severity == "ERROR":
                security_level = SecurityLevel.DANGEROUS
            elif severity == "WARNING" and security_level == SecurityLevel.SAFE:
                security_level = SecurityLevel.WARNING
        
        # Calculate security score
        score = self._calculate_security_score(issues)
        
        # Determine if safe to execute
        safe_to_execute = (security_level in [SecurityLevel.SAFE, SecurityLevel.WARNING] and 
                         score >= 70)
        
        return {
            "level": security_level,
            "issues": issues,
            "score": score, 
            "safe_to_execute": safe_to_execute,
            "tool": "semgrep",
            "recommendations": self._generate_enhanced_recommendations(issues)
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(js_file):
            os.unlink(js_file)

def run_enhanced_semgrep_analysis(self, js_file_path: str) -> dict:
    """Enhanced Semgrep analysis with more verbose output"""
    try:
        # Create custom rules file
        custom_rules_file = self.create_custom_rules_file()
        
        # Prepare enhanced Semgrep command
        cmd = [
            'semgrep',
            '--config', custom_rules_file,
            '--json',
            '--verbose',  # More detailed output
            '--no-git-ignore',
            '--disable-version-check',
            '--max-lines-per-finding', '10'  # Show more context
        ]
        
        # Add official rulesets
        for ruleset in self.security_rulesets[:3]:  # More rulesets for better coverage
            cmd.extend(['--config', ruleset])
        
        cmd.append(js_file_path)
        
        # Run Semgrep with extended timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        # Clean up
        os.unlink(custom_rules_file)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Semgrep returned code {result.returncode}")
            if result.stderr:
                print(f"Semgrep stderr: {result.stderr}")
            return {"results": []}
            
    except subprocess.TimeoutExpired:
        print("Semgrep analysis timed out")
        return {"results": []}
    except json.JSONDecodeError as e:
        print(f"Failed to parse Semgrep output: {e}")
        return {"results": []}
    except Exception as e:
        print(f"Semgrep analysis failed: {e}")
        return {"results": []}

def _generate_enhanced_recommendations(self, issues: list[dict]) -> list[str]:
    """Generate detailed security recommendations with explanations"""
    recommendations = []
    
    if not issues:
        recommendations.append("✅ No security issues detected by Semgrep")
        return recommendations
    
    # Group issues by category
    error_count = sum(1 for issue in issues if issue.get("severity") == "ERROR")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "WARNING")
    
    if error_count > 0:
        recommendations.append(f"❌ CRITICAL: {error_count} high-risk security issue(s) detected")
        recommendations.append("   Action: Block code execution immediately")
    
    if warning_count > 0:
        recommendations.append(f"⚠️ WARNING: {warning_count} potential security issue(s) found")
        recommendations.append("   Action: Review and validate before deployment")
    
    # Analyze specific vulnerability types
    rule_ids = [issue.get("rule_id", "") for issue in issues]
    cwe_ids = []
    for issue in issues:
        cwe_ids.extend(issue.get("cwe", []))
    
    # Provide specific remediation advice
    if any("innerHTML" in rule_id or "xss" in rule_id.lower() for rule_id in rule_ids):
        recommendations.append("🔧 XSS Prevention:")
        recommendations.append("   - Use textContent instead of innerHTML")
        recommendations.append("   - Sanitize all user input with DOMPurify")
        recommendations.append("   - Implement Content Security Policy (CSP)")
    
    if any("eval" in rule_id or "function" in rule_id.lower() for rule_id in rule_ids):
        recommendations.append("🔧 Code Injection Prevention:")
        recommendations.append("   - Never use eval() or Function() constructor")
        recommendations.append("   - Use JSON.parse() for data parsing")
        recommendations.append("   - Implement strict input validation")
    
    # Add CWE-specific recommendations
    if "CWE-79" in cwe_ids:
        recommendations.append("📚 Reference: CWE-79 Cross-site Scripting (XSS)")
    if "CWE-95" in cwe_ids:
        recommendations.append("📚 Reference: CWE-95 Code Injection")
    
    return recommendations

# Test function to demonstrate enhanced output
def test_enhanced_security():
    """Test enhanced security reporting"""
    from semgrep_validator import SemgrepValidator
    
    validator = SemgrepValidator()
    
    # Replace the method with enhanced version
    validator.validate_code_security = enhanced_validate_code_security.__get__(validator, SemgrepValidator)
    validator.run_semgrep_analysis = run_enhanced_semgrep_analysis.__get__(validator, SemgrepValidator)
    validator._generate_recommendations = _generate_enhanced_recommendations.__get__(validator, SemgrepValidator)
    
    # Test with problematic code
    dangerous_code = "document.getElementById('content').innerHTML = userInput + '<script>alert(1)</script>';"
    
    print("Testing enhanced security analysis:")
    result = validator.validate_code_security(dangerous_code)
    
    # Display results
    display_detailed_security_info(None, result, dangerous_code)

if __name__ == "__main__":
    test_enhanced_security()