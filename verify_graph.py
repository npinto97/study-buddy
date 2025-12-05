import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from study_buddy.agent import compiled_graph
    print("Graph loaded successfully!")
    print(compiled_graph.get_graph().draw_ascii())
except Exception as e:
    print(f"Error loading graph: {e}")
    import traceback
    traceback.print_exc()
