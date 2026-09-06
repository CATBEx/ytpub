"""
Standalone GCP/Vertex ADC sanity check — confirms auth + project/location
are correct before Stage 6 needs them, independent of any pipeline run.

Usage:
    venv\\Scripts\\activate
    python test_gcp_auth.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

print(f"GCP_PROJECT_ID: {settings.GCP_PROJECT_ID!r}")
print(f"GCP_LOCATION:   {settings.GCP_LOCATION!r}")
print(f"GEMINI_MODEL:   {settings.GEMINI_MODEL!r}")

if not settings.GCP_PROJECT_ID:
    print("\n[FAIL] GCP_PROJECT_ID is empty. Set it in config/settings.py first.")
    sys.exit(1)

print("\nConnecting via Vertex ADC (no API key)...")
try:
    from google import genai

    client = genai.Client(
        vertexai=True,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
    )
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents="Reply with exactly one word: OK",
    )
    print(f"\n[OK] Vertex ADC works. Model replied: {response.text.strip()!r}")
except Exception as e:
    print(f"\n[FAIL] {type(e).__name__}: {e}")
    print("\nCommon causes:")
    print("  - 'gcloud auth application-default login' not run, or credentials expired")
    print("  - GCP_PROJECT_ID doesn't match your actual project ID")
    print("  - Vertex AI API not enabled for that project")
    sys.exit(1)
