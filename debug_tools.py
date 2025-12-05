import sys
import os
import traceback

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Importing tools...")
    from study_buddy.utils.tools import get_all_tools
    
    print("Getting all tools...")
    tools = get_all_tools()
    print(f"Got {len(tools)} tools.")
    
    from langgraph.prebuilt import ToolNode
    print("Initializing ToolNode for each tool...")
    
    for i, tool in enumerate(tools):
        try:
            print(f"Testing tool {i}: {tool.name}")
            tn = ToolNode([tool])
            print(f"Tool {tool.name} passed.")
        except Exception as e:
            print(f"FAILED tool {tool.name}: {e}")
            # traceback.print_exc()

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
