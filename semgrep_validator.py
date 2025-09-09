#!/usr/bin/env python3

import subprocess
import json
import tempfile
import os
import re
from enum import Enum
from typing import Dict, List, Any

class SecurityLevel(Enum):
    """Security levels for code validation"""
    SAFE = "SAFE"
    WARNING = "WARNING" 
    DANGEROUS = "DANGEROUS"
    BLOCKED = "BLOCKED"

class SemgrepValidator:
    """Semgrep-based security validator for JavaScript code"""
    
    def __init__(self):
        """Initialize Semgrep validator"""
        self.semgrep_available = self._check_semgrep_availability()
        self.security_rulesets = [
            "p/javascript",
            "p/security-audit", 
            "p/owasp-top-ten"
        ]
        
        # Fallback patterns if Semgrep not available
        self.dangerous_patterns = {
            'eval_usage': {
                'pattern': r'\beval\s*\(',
                'message': 'Use of eval() is dangerous - allows code injection',
                'severity': 'ERROR',
                'cwe': ['CWE-95']
            },
            'function_constructor': {
                'pattern': r'\bFunction\s*\(',
                'message': 'Function constructor can execute arbitrary code',
                'severity': 'ERROR', 
                'cwe': ['CWE-95']
            },
            'innerHTML_injection': {
                'pattern': r'\.innerHTML\s*=.*[\+\$]',
                'message': 'innerHTML with concatenation may cause XSS',
                'severity': 'WARNING',
                'cwe': ['CWE-79']
            },
            'document_write': {
                'pattern': r'\bdocument\.write\s*\(',
                'message': 'document.write is deprecated and unsafe',
                'severity': 'WARNING',
                'cwe': ['CWE-79']
            },
            'onclick_attribute': {
                'pattern': r'\.onclick\s*=',
                'message': 'Use addEventListener instead of onclick attribute',
                'severity': 'INFO',
                'cwe': []
            }
        }
        
        print(f" Semgrep Validator initialized")
        print(f"   Semgrep Available: {'✅' if self.semgrep_available else '❌'}")
        if not self.semgrep_available:
            print("   Using fallback pattern matching")
    
    def _check_semgrep_availability(self) -> bool:
        """Check if Semgrep is available on the system"""
        try:
            result = subprocess.run(['semgrep', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def create_temp_js_file(self, code: str) -> str:
        """Create temporary JavaScript file for analysis"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            return f.name
    
    def run_semgrep_analysis(self, js_file_path: str) -> Dict[str, Any]:
        """Run Semgrep security analysis"""
        if not self.semgrep_available:
            return {"results": []}
        
        try:
            cmd = [
                'semgrep',
                '--config', 'p/javascript',
                '--config', 'p/security-audit',
                '--json',
                '--no-git-ignore',
                '--disable-version-check',
                js_file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print(f"Semgrep error: {result.stderr}")
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
    
    def fallback_pattern_analysis(self, code: str) -> List[Dict[str, Any]]:
        """Fallback pattern-based security analysis"""
        issues = []
        
        for rule_name, rule_info in self.dangerous_patterns.items():
            pattern = rule_info['pattern']
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append({
                    "type": "PATTERN_MATCH",
                    "rule_id": rule_name,
                    "message": rule_info['message'],
                    "severity": rule_info['severity'],
                    "line": line_num,
                    "column": match.start() - code.rfind('\n', 0, match.start()),
                    "code": match.group(0),
                    "cwe": rule_info.get('cwe', []),
                    "confidence": "MEDIUM"
                })
        
        return issues
    
    def _calculate_security_score(self, issues: List[Dict]) -> int:
        """Calculate security score based on issues"""
        if not issues:
            return 100
        
        score = 100
        for issue in issues:
            severity = issue.get('severity', 'INFO')
            if severity == 'ERROR':
                score -= 30
            elif severity == 'WARNING':
                score -= 15
            elif severity == 'INFO':
                score -= 5
        
        return max(0, score)
    
    def _determine_security_level(self, issues: List[Dict]) -> SecurityLevel:
        """Determine overall security level"""
        if not issues:
            return SecurityLevel.SAFE
        
        has_error = any(issue.get('severity') == 'ERROR' for issue in issues)
        has_warning = any(issue.get('severity') == 'WARNING' for issue in issues)
        
        if has_error:
            return SecurityLevel.DANGEROUS
        elif has_warning:
            return SecurityLevel.WARNING
        else:
            return SecurityLevel.SAFE
    
    def validate_code_security(self, code: str) -> Dict[str, Any]:
        """Main security validation method"""
        if not code or not code.strip():
            return {
                "level": SecurityLevel.SAFE,
                "issues": [],
                "score": 100,
                "safe_to_execute": True,
                "tool": "validator"
            }
        
        if self.semgrep_available:
            # Use Semgrep analysis
            js_file = self.create_temp_js_file(code)
            
            try:
                semgrep_result = self.run_semgrep_analysis(js_file)
                issues = []
                
                for finding in semgrep_result.get("results", []):
                    extra = finding.get("extra", {})
                    issues.append({
                        "type": "SEMGREP_FINDING",
                        "rule_id": finding.get("check_id", "unknown"),
                        "message": extra.get("message", "Security issue detected"),
                        "severity": extra.get("severity", "INFO"),
                        "line": finding.get("start", {}).get("line", 0),
                        "column": finding.get("start", {}).get("col", 0),
                        "code": extra.get("lines", ""),
                        "confidence": extra.get("confidence", "MEDIUM")
                    })
                
            finally:
                if os.path.exists(js_file):
                    os.unlink(js_file)
        else:
            # Use fallback pattern analysis
            issues = self.fallback_pattern_analysis(code)
        
        # Calculate results
        security_level = self._determine_security_level(issues)
        score = self._calculate_security_score(issues)
        safe_to_execute = security_level in [SecurityLevel.SAFE, SecurityLevel.WARNING] and score >= 70
        
        return {
            "level": security_level,
            "issues": issues,
            "score": score,
            "safe_to_execute": safe_to_execute,
            "tool": "semgrep" if self.semgrep_available else "fallback"
        }
    
    def get_security_summary(self, code: str) -> str:
        """Get a brief security summary"""
        result = self.validate_code_security(code)
        level = result['level']
        score = result['score']
        issue_count = len(result['issues'])
        
        if level == SecurityLevel.SAFE:
            return f"✅ SAFE ({score}/100)"
        elif level == SecurityLevel.WARNING:
            return f"⚠️ WARNING ({score}/100, {issue_count} issues)"
        elif level == SecurityLevel.DANGEROUS:
            return f"❌ DANGEROUS ({score}/100, {issue_count} issues)"
        else:
            return f"⛔ BLOCKED"

def test_validator():
    """Test the validator with sample code"""
    validator = SemgrepValidator()
    
    test_cases = [
        # Safe code
        "document.getElementById('test').textContent = 'Hello';",
        
        # Dangerous code
        "eval('alert(1)');",
        "document.getElementById('content').innerHTML = userInput;",
        "Function('alert(1)')();"
    ]
    
    print("\n🧪 Testing Semgrep Validator")
    print("=" * 40)
    
    for i, code in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {code}")
        result = validator.validate_code_security(code)
        print(f"   Level: {result['level'].value}")
        print(f"   Score: {result['score']}/100")
        print(f"   Safe: {'Yes' if result['safe_to_execute'] else 'No'}")
        
        if result['issues']:
            print(f"   Issues: {len(result['issues'])}")
            for issue in result['issues'][:2]:  # Show first 2
                print(f"     - {issue.get('message', 'Unknown issue')}")

if __name__ == "__main__":
    test_validator()