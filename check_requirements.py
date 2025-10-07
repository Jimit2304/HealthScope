#!/usr/bin/env python3
"""
Check if all requirements are met before running the application
"""

import os
import sys

def check_requirements():
    """Check all requirements for the application"""
    issues = []
    
    # Check Python packages
    required_packages = [
        'flask', 'flask_cors', 'mysql.connector', 
        'numpy', 'pandas', 'sklearn', 'werkzeug'
    ]
    
    print("Checking Python packages...")
    for package in required_packages:
        try:
            if package == 'mysql.connector':
                import mysql.connector
            elif package == 'flask_cors':
                import flask_cors
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            issues.append(f"Install {package}")
    
    # Check dataset file
    print("\nChecking dataset file...")
    dataset_path = "diabetes[1].csv"
    if os.path.exists(dataset_path):
        print(f"✓ {dataset_path}")
    else:
        print(f"✗ {dataset_path} - MISSING")
        issues.append("Dataset file 'diabetes[1].csv' not found")
    
    # Check MySQL connection
    print("\nChecking MySQL connection...")
    try:
        import mysql.connector
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            port=3306
        )
        connection.close()
        print("✓ MySQL connection")
    except Exception as e:
        print(f"✗ MySQL connection - {e}")
        issues.append("MySQL connection failed")
    
    # Summary
    print("\n" + "="*50)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix these issues before running the application.")
        return False
    else:
        print("✅ ALL REQUIREMENTS MET!")
        print("You can now run: python app.py")
        return True

if __name__ == "__main__":
    check_requirements()