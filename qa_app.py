# qa_app.py

import torch
from transformers import pipeline

def main():
    """
    Main function to run the interactive Question-Answering application.
    """
    # Check if a GPU is available and set the device
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # 1. Initialize the question-answering pipeline
    # This will download the model on first run.
    # We specify the device to use GPU if available.
    try:
        qa_pipeline = pipeline("question-answering", device=device)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Please ensure you have a working internet connection and PyTorch is installed correctly.")
        return

    # 2. Start the interactive loop
    print("\n--- Interactive Question-Answering Bot ---")
    print("I will find answers to your questions from the text you provide.")
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 50)

    # Get the context from the user
    print("First, paste your context text. When you are done, type 'ENDOFTEXT' on a new line and press Enter.")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'ENDOFTEXT':
            break
        lines.append(line)
    context = "\n".join(lines)

    print("\n Context received! You can now start asking questions.")
    print("-" * 50)

    # Loop for asking questions
    while True:
        question = input("\nEnter your question: ")

        if question.lower() in ['quit', 'exit']:
            print("\nEnding session. Goodbye!")
            break

        if not question.strip():
            print("Please enter a question.")
            continue

        result = qa_pipeline(question=question, context=context)

        print(f"\nAnswer: '{result['answer']}'")
        print(f"Confidence: {result['score']:.2%}") # Format score as a percentage
        print("-" * 50)


if __name__ == "__main__":
    main()