from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# -----------------------------
# LLM (local via Ollama)
# -----------------------------
llm = OllamaLLM(
    model="qwen:7b",
    temperature=0.2
)

# -----------------------------
# Memory (list of messages)
# -----------------------------
chat_history = []

# -----------------------------
# Travel keyword guard
# -----------------------------
travel_terms = {
    "flight", "flights", "hotel", "hotels", "destination", "destinations",
    "trip", "trips", "travel", "tour", "tourism", "itinerary", "vacation",
    "accommodation", "booking", "passport", "visa", "airfare", "layover",
    "sightseeing", "cruise", "resort", "rental", "transportation", "airport"
}

def is_travel_query(text):
    """Return True if the user query looks travel-related."""
    normalized = text.lower()
    return any(term in normalized for term in travel_terms)

# -----------------------------
# Prompt template
# -----------------------------
prompt_template = PromptTemplate(
    input_variables=["history", "user_input"],
    template="""
You are a Travel Agent chatbot.

Rules:
- Answer ONLY travel-related questions (flights, hotels, destinations, trips, travel tips).
- If question is NOT travel-related, respond EXACTLY:
I can’t help with it.

Conversation so far:
{history}

User: {user_input}
Assistant:
"""
)

# -----------------------------
# Helper: format history
# -----------------------------
def format_history(history):
    return "\n".join(history)

# -----------------------------
# Summarization function
# -----------------------------
def summarize_history(history):
    """Summarize the stored conversation so we can reduce context size."""
    summary_prompt = f"""
Summarize this conversation briefly:

{format_history(history)}
"""
    return llm.invoke(summary_prompt)

# -----------------------------
# Streaming response (manual)
# -----------------------------
def stream_response(prompt):
    """Stream the LLM output token-by-token and return the full text."""
    response = llm.stream(prompt)
    full_output = ""

    for chunk in response:
        print(chunk, end="", flush=True)
        full_output += chunk

    print("\n")
    return full_output

# -----------------------------
# Chat loop
# -----------------------------
def chat():
    print("🌍 Travel Agent Chatbot (type 'exit' to quit)\n")

    conversation_count = 0

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Always keep all messages in the Python list memory.
        chat_history.append(f"User: {user_input}")

        # If the query is not travel-related, return the exact canned response.
        if not is_travel_query(user_input):
            response = "I can’t help with it."
            print(f"Agent: {response}\n")
            chat_history.append(f"Assistant: {response}")
        else:
            formatted_prompt = prompt_template.format(
                history=format_history(chat_history),
                user_input=user_input
            )

            print("Agent: ", end="")
            response = stream_response(formatted_prompt)
            chat_history.append(f"Assistant: {response}")

        conversation_count += 1

        # -----------------------------
        # Summarize after every 5 conversations
        # -----------------------------
        if conversation_count % 5 == 0:
            print("🔄 Summarizing conversation...\n")

            summary = summarize_history(chat_history)

            # Replace older messages with the summary to reduce context size.
            chat_history.clear()
            chat_history.append(f"Summary: {summary}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    chat()