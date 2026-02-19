# Multi-LLM Orchestration Platform

A production-ready FastAPI application that orchestrates multiple Large Language Models (LLMs) with intelligent response grading and comparison, featuring a React-based web interface.

## Features

- **Multi-LLM Support**: Currently integrates GPT-5.2 and PHI-4 mini reasoning models
- **Intelligent Orchestration**: Automatically sends prompts to multiple models and compares responses
- **Response Grading**: AI-powered evaluation and ranking of model responses
- **React Web Interface**: Interactive frontend for easy prompt submission and response comparison
- **Production Deployment**: Containerized and deployed on Azure Container Instances
- **Comprehensive Logging**: Full tracing and debugging capabilities
- **API Endpoints**: RESTful API for programmatic access

## Architecture

```
Frontend (React + Vite) → FastAPI Server → Azure OpenAI Services
                                        ├── GPT-5.2 Chat
                                        └── PHI-4 Mini Reasoning
```

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd MultiLLMApp
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file with your Azure OpenAI credentials:
   ```
   GPT52_ENDPOINT=your-gpt52-endpoint
   GPT52_KEY=your-gpt52-key
   GPT52_DEPLOYMENT=gpt-5.2-chat
   GPT52_API_VERSION=2024-12-01-preview
   
   PHI4_ENDPOINT=your-phi4-endpoint
   PHI4_KEY=your-phi4-key
   PHI4_DEPLOYMENT=phi-4-mini-reasoning
   PHI4_API_VERSION=2024-05-01-preview
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Access the application**
   - Web Interface: http://localhost:8087
   - API Documentation: http://localhost:8087/docs

### Using Docker

1. **Build the container**
   ```bash
   docker build -t multi-llm-app .
   ```

2. **Run the container**
   ```bash
   docker run -p 8087:8000 --env-file .env multi-llm-app
   ```

## API Endpoints

- `GET /` - React web interface
- `POST /api/llm` - Multi-LLM orchestration endpoint
- `POST /ask` - Alternative endpoint for compatibility
- `GET /status` - API status information
- `POST /health` - Health check with environment validation

### Example API Usage

```bash
curl -X POST "http://localhost:8087/api/llm" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain quantum computing"}'
```

Response format:
```json
{
  "prompt": "Explain quantum computing",
  "answers": {
    "gpt-5.2-chat": "Quantum computing response...",
    "phi-4-mini-reasoning": "PHI-4 response..."
  },
  "best": "gpt-5.2-chat",
  "grades": {
    "gpt-5.2-chat": 8.5,
    "phi-4-mini-reasoning": 7.2
  }
}
```

## Deployment

The application is production-ready and deployed on Azure Container Instances:

🔗 **Live Demo**: [llm-multi-llm-api.centralindia.azurecontainer.io:8000](http://llm-multi-llm-api.centralindia.azurecontainer.io:8000)

### Azure Container Deployment

```bash
# Build and push to Azure Container Registry
docker build -t llm-orchestration-app .
docker tag llm-orchestration-app:latest your-registry.azurecr.io/llm-orchestration-app:latest
docker push your-registry.azurecr.io/llm-orchestration-app:latest

# Deploy to Azure Container Instances
az container create \
  --resource-group your-rg \
  --name llm-multi-llm-api \
  --image your-registry.azurecr.io/llm-orchestration-app:latest \
  --environment-variables [your-env-vars]
```

## Features Deep Dive

### Multi-LLM Orchestration
- Parallel API calls to multiple models
- Timeout handling and error recovery
- Response normalization and formatting

### Intelligent Response Grading
- Content quality analysis
- Response length evaluation
- Best response selection algorithm

### Comprehensive Logging
- Request/response tracing
- Performance monitoring
- Error debugging with stack traces

### React Frontend
- Real-time model comparison
- Interactive prompt interface
- Response grading visualization

## Development

### Project Structure
```
MultiLLMApp/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── static/             # React build files
├── frontend/           # React source code
└── README.md          # This file
```

### Adding New Models

1. Update the `query_llm` function in `main.py`
2. Add environment variable configuration
3. Update the models list in the orchestration logic

## Troubleshooting

### Common Issues

1. **Empty GPT-5.2 Responses**: Ensure `max_completion_tokens` is set to 1024+ for reasoning models
2. **Static Files Not Loading**: Verify static file mounting in FastAPI configuration
3. **API Rate Limits**: Check Azure OpenAI quota and rate limits

### Debug Logging

Enable detailed logging to trace API calls:
```python
# Already configured in main.py with comprehensive logging
```

View container logs in Azure:
```bash
az container logs --name llm-multi-llm-api --resource-group your-rg
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI and React
- Powered by Azure OpenAI Services
- Deployed on Azure Container Instances