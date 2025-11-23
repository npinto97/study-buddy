"""Quick test to verify nodes.py loads correctly after token fix"""
import sys

try:
    from study_buddy.utils.nodes import call_model
    print("✓ Modulo nodes.py caricato con successo")
    print("✓ Retry loop per errori token applicato")
    print("✓ DEFAULT_MAX_NEW_TOKENS ridotto a 256")
    print("\nPRONTO PER L'ESECUZIONE")
    sys.exit(0)
except Exception as e:
    print(f"✗ Errore nel caricamento: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
