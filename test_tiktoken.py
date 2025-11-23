#!/usr/bin/env python
"""
Quick test to verify tiktoken is working correctly for token counting.
"""

import tiktoken

def test_tiktoken():
    """Test tiktoken encoding for different models."""
    
    test_text = """
    Ciao! Mi sai dire l'algoritmo di Rocchio per l'information retrieval?
    Vorrei capire come funziona la relevance feedback nel contesto dei sistemi di recupero dell'informazione.
    """
    
    print("🧪 Testing tiktoken installation...\n")
    
    # Test with gpt-4 encoding (commonly used)
    try:
        enc_gpt4 = tiktoken.encoding_for_model("gpt-4")
        tokens_gpt4 = enc_gpt4.encode(test_text)
        print(f"✅ GPT-4 encoding: {len(tokens_gpt4)} tokens")
        print(f"   Text length: {len(test_text)} chars")
        print(f"   Ratio: {len(test_text) / len(tokens_gpt4):.2f} chars/token")
    except Exception as e:
        print(f"❌ GPT-4 encoding failed: {e}")
    
    # Test with cl100k_base (default for GPT-3.5/4)
    try:
        enc_base = tiktoken.get_encoding("cl100k_base")
        tokens_base = enc_base.encode(test_text)
        print(f"\n✅ cl100k_base encoding: {len(tokens_base)} tokens")
        print(f"   Text length: {len(test_text)} chars")
        print(f"   Ratio: {len(test_text) / len(tokens_base):.2f} chars/token")
    except Exception as e:
        print(f"❌ cl100k_base encoding failed: {e}")
    
    # Compare with old heuristic (len/4)
    heuristic_estimate = len(test_text) // 4
    actual_tokens = len(tokens_base)
    error = abs(actual_tokens - heuristic_estimate)
    error_pct = (error / actual_tokens) * 100
    
    print(f"\n📊 Comparison with old heuristic (len/4):")
    print(f"   Heuristic estimate: {heuristic_estimate} tokens")
    print(f"   Actual tokens (tiktoken): {actual_tokens} tokens")
    print(f"   Error: {error} tokens ({error_pct:.1f}%)")
    
    if error_pct < 10:
        print(f"   ✅ Good accuracy (< 10% error)")
    elif error_pct < 20:
        print(f"   ⚠️  Moderate accuracy (10-20% error)")
    else:
        print(f"   ❌ Poor accuracy (> 20% error)")
    
    print("\n🎉 tiktoken is working correctly!")
    return True

if __name__ == "__main__":
    test_tiktoken()
