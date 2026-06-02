# 🎭 Interactive Fiction Agentic System

A state-of-the-art multi-agent framework powered by **Pydantic AI** and **MGraph DB** to simulate, build, and interact with dynamic fictional worlds. The system extracts entities and relationships from a base story to build a multi-dimensional Knowledge Graph, then orchestrates a team of specialized AI agents to power a live, responsive, choice-driven roleplaying experience.

---

## 🚀 Key Features

* **Dual-Phase Architecture**: Seamlessly transitions from static plot analysis to interactive gameplay simulation.
* **Knowledge Graph Extraction (Phase 1)**: Employs dedicated Named Entity Recognition (NER) and relationship mapping agents to parse narratives into structured subject-predicate-object triplets.
* **Interactive Story Engine (Phase 2)**:
  * **Avatar Agents**: Represent unique characters in the universe, autonomously deciding how they react to player choices and writing their own dialogue.
  * **Routing Agent**: Smarter classification dividing player actions from factual lore queries.
  * **Informational RAG Agent**: Leverages the knowledge graph as a retrieval-augmented source to answer complex questions about the world and relationships.
  * **Guardian Agent**: Safety and alignment guardrails to evaluate and filter inappropriate prompts.
  * **Story Teller Agent**: Compiles multi-character outputs into a rich, coherent literary narrative written from the protagonist’s perspective.
* **Serialized Graph DB**: Implements `mgraph_db` to save/restore the world structure as `knowledge_graph.json`.

---

## 🛠️ Technology Stack

* **Language**: Python 3.10+
* **LLM Engine**: Anthropic Claude Haiku 4.5
* **Agent Framework**: [Pydantic AI](https://github.com/pydantic/pydantic-ai) for structured outputs, prompt templates, and type safety.
* **Database**: `mgraph_db` & `osbot_utils` for advanced schema-enforced graph modeling.
* **Environment**: `python-dotenv` for key management.

---

## 📦 Installation & Setup

Ensure you have Python 3.10 or newer installed. Follow these steps to set up and run the project:

### 1. Set Up a Virtual Environment

Navigate to the project root and create a virtual environment to isolate dependencies:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install the optimized list of libraries required to run the script:

```bash
pip install -r requirements.txt
```

### 3. Configure the Environment

Create a `.env` file in the root of the project to securely store your API keys:

```env
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

---

## 🎮 How to Run

### Play the Interactive Game
Start the simulation immediately using the pre-compiled `knowledge_graph.json` database:

```bash
python agentic_system.py
```

### Reconstruct the World Database
If you modify `plot.md` or want to regenerate the knowledge graph from scratch using the extraction agents, run:

```bash
python agentic_system.py --construct_graph
```

### CLI Command Options

Configure the engine using command-line arguments:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--construct_graph` | Switch | `False` | Run the extraction agents (Phase 1) to rebuild the Knowledge Graph from `plot.md`. |
| `--plot_file` | `str` | `plot.md` | Path to the base markdown story file to extract. |
| `--graph_file` | `str` | `knowledge_graph.json` | Destination/source path for the serialized JSON graph database. |
| `--story_file` | `str` | `my_story.txt` | File path where the generated story log will be continually appended. |

---

## 👥 Meet the Cast (Zania's Locket Demo)

The included default narrative revolves around **Zania's Locket**, featuring:
1. **Zania Sagan**: A custodian at the Hospice who holds a mysterious locket.
2. **Samuel**: A guiding voice on the other end of the locket.
3. **Sasha**: A child-brain-in-a-jar consciousness controlling the facility.
4. **Grandmother**: The dark architect of the Hospice and its biological "Soup".
