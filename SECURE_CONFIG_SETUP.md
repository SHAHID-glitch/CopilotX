# 🔐 Secure API Configuration Setup

## Overview
CopilotX now uses environment variables to securely manage API keys. This prevents accidental exposure of sensitive credentials.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install `python-dotenv` which manages environment variables from `.env` files.

### 2. Configure Your API Keys
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your actual API keys:
   ```
   COPILOTX_API_KEY=your_actual_key_here
   AZURE_OPENAI_KEY=your_actual_key_here
   OPENAI_API_KEY=your_actual_key_here
   # ... etc
   ```

### 3. Security Notes
- ✅ The `.env` file is already in `.gitignore` and will NOT be committed to GitHub
- ✅ The `.env.example` file serves as a template and is safe to commit
- ✅ All secrets are loaded at runtime from environment variables
- ✅ Never hardcode API keys in source code

### 4. How It Works
The `config.py` now:
1. Loads environment variables from `.env` file via `python-dotenv`
2. Falls back to `config.json` if environment variables are not set
3. Uses placeholders if neither is available

### 5. Using in Your Code
```python
from config import CopilotXConfig

config = CopilotXConfig()
api_key = config.api_keys.get("copilotx_primary")
```

## Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `COPILOTX_API_KEY` | Primary CopilotX API Key | Yes |
| `AZURE_OPENAI_KEY` | Azure OpenAI Service Key | Optional |
| `AZURE_COGNITIVE_KEY` | Azure Cognitive Services Key | Optional |
| `OPENAI_API_KEY` | OpenAI API Key | Optional |
| `ANTHROPIC_API_KEY` | Anthropic API Key | Optional |
| `GOOGLE_AI_KEY` | Google AI API Key | Optional |
| `ENVIRONMENT` | Deployment environment | No (default: production) |
| `DEBUG_MODE` | Enable debug logging | No (default: false) |

## Verification
To verify your setup is correct, run:
```bash
python -c "from config import CopilotXConfig; c = CopilotXConfig(); print('✅ Configuration loaded successfully!')"
```

If you see `✅ Configuration loaded successfully!`, your setup is complete.
