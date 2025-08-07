# Extractive Question-Answering Bot with Transformers

A Python application that uses a state-of-the-art Transformer model from Hugging Face to perform extractive question answering. The user provides a block of text (context) and can then ask questions to receive answers extracted directly from that text.


## Features

-   **Interactive CLI:** A user-friendly command-line interface to input context and ask multiple questions.
-   **State-of-the-Art NLP Model:** Powered by a pre-trained Transformer model (DistilBERT) from the Hugging Face library.
-   **Extractive QA:** Finds and highlights the exact span of text that answers a given question.
-   **GPU Accelerated:** Automatically utilizes a GPU if available (via PyTorch) for faster performance.

## Tech Stack

-   Python 3
-   Hugging Face `transformers`
-   PyTorch

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main application script from your terminal:

```bash
python qa_app.py
```

The application will first prompt you to paste a context text. After you provide the context, you can begin asking questions about it. Type `quit` or `exit` to end the session.
