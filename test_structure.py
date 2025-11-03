"""
Test script to validate code structure without requiring all dependencies.
"""

import ast
import os

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_file_structure():
    """Verify all required files exist."""
    required_files = [
        'data_fetcher.py',
        'feature_engineering.py',
        'models.py',
        'evaluation.py',
        'visualization.py',
        'experiment.py',
        'requirements.txt',
        'README.md',
        '.gitignore'
    ]
    
    print("="*80)
    print("CHECKING FILE STRUCTURE")
    print("="*80)
    
    all_exist = True
    for filename in required_files:
        exists = os.path.exists(filename)
        status = "✓" if exists else "✗"
        print(f"{status} {filename}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_python_files():
    """Check syntax of all Python files."""
    python_files = [
        'data_fetcher.py',
        'feature_engineering.py',
        'models.py',
        'evaluation.py',
        'visualization.py',
        'experiment.py'
    ]
    
    print("\n" + "="*80)
    print("CHECKING PYTHON SYNTAX")
    print("="*80)
    
    all_valid = True
    for filename in python_files:
        if os.path.exists(filename):
            valid, message = check_python_syntax(filename)
            status = "✓" if valid else "✗"
            print(f"{status} {filename}: {message}")
            if not valid:
                all_valid = False
    
    return all_valid

def check_module_structure():
    """Check if modules have expected classes and functions."""
    print("\n" + "="*80)
    print("CHECKING MODULE STRUCTURE")
    print("="*80)
    
    checks = [
        ('data_fetcher.py', ['BTCDataFetcher', 'save_data', 'load_data']),
        ('feature_engineering.py', ['FeatureEngineer']),
        ('models.py', ['BaseModel', 'LSTMModel', 'GRUModel', 'TransformerModel', 
                       'RandomForestModel', 'XGBoostModel']),
        ('evaluation.py', ['ModelEvaluator', 'BacktestEvaluator']),
        ('visualization.py', ['Visualizer']),
        ('experiment.py', ['BTCPredictionExperiment', 'main'])
    ]
    
    all_valid = True
    for filename, expected_names in checks:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    code = f.read()
                tree = ast.parse(code)
                
                # Extract class and function names
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                all_names = classes + functions
                
                missing = [name for name in expected_names if name not in all_names]
                
                if missing:
                    print(f"✗ {filename}: Missing {missing}")
                    all_valid = False
                else:
                    print(f"✓ {filename}: All expected components found")
                    
            except Exception as e:
                print(f"✗ {filename}: Error checking structure: {e}")
                all_valid = False
    
    return all_valid

def count_lines():
    """Count lines of code."""
    print("\n" + "="*80)
    print("CODE STATISTICS")
    print("="*80)
    
    python_files = [
        'data_fetcher.py',
        'feature_engineering.py',
        'models.py',
        'evaluation.py',
        'visualization.py',
        'experiment.py'
    ]
    
    total_lines = 0
    for filename in python_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = len(f.readlines())
            total_lines += lines
            print(f"  {filename}: {lines} lines")
    
    print(f"\nTotal: {total_lines} lines of Python code")

def main():
    """Run all checks."""
    print("\nBTC PREDICTION EXPERIMENT - CODE VALIDATION")
    print("="*80 + "\n")
    
    results = []
    
    # Check file structure
    results.append(("File Structure", check_file_structure()))
    
    # Check Python syntax
    results.append(("Python Syntax", check_python_files()))
    
    # Check module structure
    results.append(("Module Structure", check_module_structure()))
    
    # Count lines
    count_lines()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Code structure is valid!")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
    print("="*80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
