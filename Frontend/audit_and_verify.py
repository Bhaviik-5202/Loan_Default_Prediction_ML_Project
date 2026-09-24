"""
Audit & Verification runner for when the terminal current working directory is Frontend/.
"""
import os
import sys

_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_FRONTEND_DIR)

# Delegate to the root audit_and_verify.py
sys.path.insert(0, _PROJECT_ROOT)
import audit_and_verify

if __name__ == "__main__":
    audit_and_verify.audit_ml_artifacts()
    audit_and_verify.audit_ml_inference()
    audit_and_verify.audit_flask_routes_and_apis()
    audit_and_verify.audit_notebook()
    audit_and_verify.audit_sop_compliance()
    print("\n" + "="*65)
    print("ALL AUDIT PHASES PASSED — WEEKS 1-10 FULLY VERIFIED!")
    print("="*65 + "\n")
