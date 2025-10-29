# Demo - EverMemOS Interactive Examples

[English](README.md) | [简体中文](README_zh.md)

This directory contains interactive demos showcasing the core functionality of EverMemOS.

## 📂 Contents

### Core Demo Scripts

- **`extract_memory.py`** - Memory extraction from conversation data
  - Processes conversation files from the `data/` directory
  - Extracts MemCells and generates user profiles
  - Saves results to configured database (MongoDB) and local outputs

- **`chat_with_memory.py`** - Interactive chat with memory-enhanced AI
  - Command-line interface for conversing with AI agents
  - Leverages extracted memories for context-aware responses
  - Demonstrates end-to-end memory retrieval and usage

### Configuration Files

- **`memory_config.py`** - Memory system configuration
- **`memory_utils.py`** - Utility functions for memory operations
- **`i18n_texts.py`** - Internationalization text resources

### Output Directory

- **`chat_history/`** - Saved chat conversation logs
- **`memcell_outputs/`** - Extracted MemCell outputs (auto-generated)

## 🚀 Quick Start

### 1. Extract Memories

First, extract memories from sample conversation data:

```bash
uv run python src/bootstrap.py demo/extract_memory.py
```

This will:
- Read conversation data from `data/` directory
- Extract MemCells using LLM
- Generate user profiles
- Store in MongoDB and save to local files

### 2. Chat with Memory

After extraction, start an interactive chat session:

```bash
uv run python src/bootstrap.py demo/chat_with_memory.py
```

Features:
- Real-time conversation with AI
- Automatic memory retrieval based on context
- Bilingual support (English/Chinese)
- Chat history auto-save

## 📝 Usage Tips

### Memory Extraction

The extraction script processes JSON files in the `data/` directory (configured in `ExtractModeConfig` in `memory_config.py`). Ensure your conversation files follow the [GroupChatFormat](../data_format/group_chat/group_chat_format.md) specification. For convenience, we provide bilingual sample data files:
- **English versions:** [group_chat_en.json](../data/group_chat_en.json) and [assistant_chat_en.json](../data/assistant_chat_en.json)
- **Chinese versions:** [group_chat_zh.json](../data/group_chat_zh.json) and [assistant_chat_zh.json](../data/assistant_chat_zh.json)

See the [data documentation](../data/README.md) for details.

### Interactive Chat

During chat sessions:
- Type naturally to converse with the AI
- The system automatically retrieves relevant memories
- Use `exit` to end the session
- Chat history is saved automatically (defaults to loading previous 5 messages as context)

## 🔧 Configuration

Edit `memory_config.py` to customize:
- LLM model selection
- Memory extraction parameters
- Database connection settings
- Output directories

## 📊 Example Output

After running `extract_memory.py`, you'll find:

```
demo/memcell_outputs/
├── user_profiles/
│   ├── user_101_profile.json
│   └── user_102_profile.json
└── memcells/
    ├── group_001_memcells.json
    └── episode_memories.json
```

## 🔗 Related Documentation

- [Group Chat Format Specification](../data_format/group_chat/group_chat_format.md)
- [API Documentation](../docs/api_docs/agentic_v3_api.md)
- [Data Documentation](../data/README.md)
- [Internationalization Guide](../docs/dev_docs/chat_i18n_usage.md)

## 📖 Demo Data Overview

### Group Chat Scenario (group_chat_en.json / group_chat_zh.json)

**Project Context:** AI product work group documenting the complete development journey of "Smart Sales Assistant"

**Key Contents:**
- MVP development phase: RAG-based Q&A system
- Advanced feature iteration: Emotion recognition, memory system
- Team collaboration practices: Complete workflow from requirements to delivery

**Available in:** English and Chinese versions

**Good for exploring:** Team collaboration patterns, project management, technical solution evolution

### Assistant Scenario (assistant_chat_en.json / assistant_chat_zh.json)

**Conversation Context:** Personal health & lifestyle assistant documenting nearly 2 months of continuous interaction

**Key Contents:**
- Travel planning: Food recommendations, itinerary suggestions
- Health management: Weight monitoring, dietary guidance
- Exercise recovery: Training advice, post-injury rehabilitation

**Available in:** English and Chinese versions

**Good for exploring:** Personalized services, long-term memory accumulation, contextual understanding

## ❓ Recommended Questions

**Group Chat AI Scenario Examples:**
- What did Alex/Betty/... do in the emotion recognition project?
- Based on the emotion recognition project, what work capabilities does Alex/Betty/... demonstrate?
- What are the deliverable results of the emotion recognition project?
- How is the memory system project progressing?

**Assistant AI Scenario Examples:**
- Please recommend sports suitable for me.
- Please recommend food I might like.
- How is my health condition?


## 💡 Need Help?

- Check the main [README](../README.md) for setup instructions
- Review the [Batch Memorization Usage Guide](../docs/dev_docs/run_memorize_usage.md)
- Open an issue on GitHub

---

**Happy exploring! 🧠✨**

