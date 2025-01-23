from study_buddy.config import logger
from study_buddy.agents.chatbot import chatbot, chatbot_tools
from study_buddy.modules.graph_builder import build_graph, print_graph

logger.info("Starting Study Buddy Application...")


def main():
    graph = build_graph(chatbot, chatbot_tools)
    print_graph(graph)

    print("Graph built successfully!")

    # Run chatbot loop
    config = {"configurable": {"thread_id": "7"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )
        for event in events:
            # print("DEBUG: Messages:", event["messages"])
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
