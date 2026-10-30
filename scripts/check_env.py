"""Check environment: Python version, required packages, data files, outputs."""

import importlib
import os
import sys

REQUIRED = [
    "pandas", "numpy", "scipy", "sklearn", "joblib",
    "matplotlib", "fastembed", "fastapi", "uvicorn", "pydantic",
]

REQUIRED_FILES = [
    "data/train.csv",
    "data/validation.csv",
]

def main():
    ok = True
    print(f"Python: {sys.version}")

    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}  → pip install {pkg}")
            ok = False

    for f in REQUIRED_FILES:
        if os.path.exists(f):
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} missing")
            ok = False

    env_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
    for v in env_vars:
        val = os.environ.get(v)
        if val:
            print(f"  ✅ {v} (set)")
        else:
            print(f"  ⚠️  {v} (not set – needed for API/DB phase only)")

    print()
    print("PASS ✅" if ok else "FAIL ❌ – fix the items above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
