# ComFit

# Goal: Build an Enhanced Hybrid Chatbot

- This chatbot processes user queries against a given PDF using LlamaIndex, Ollama LLM, and optional Google Search.
- It applies Retrieval-Augmented Correction (RAC) to extract, verify, and correct factual claims from the LLM’s response.
- It uses Confidence Cascade to suppress or flag answers based on verification confidence.
- Supports toggleable RAC,settable verification mode (local, web, hybrid), and stats tracking.
- Embedding model (e.g., `nomic-embed-text`) is switchable via `Settings.embed_model`.
- Designed to prioritize formal correctness, avoid hallucinations, and support both exploratory and cautious answer generation.

## Getting Started

After cloning the repository, download and install the dependencies (make sure you have Node.js, npm, and python installed):

Frontend:

```bash
npm install
```

Backend:

```bash
pip install -r requirements.txt
```

Then, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

For the backend, first download the requirements with

```bash
pip install -r requirements.txt
```

Then run the server with

```bash
python main.py
```

or

```bash
python start.py
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
