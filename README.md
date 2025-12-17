# ServiceNow Ticket Analysis Agent

A Streamlit-based web application that uses GPT to intelligently categorize ServiceNow tickets. It analyzes ticket descriptions and solutions to automatically assign meaningful categories, helping identify patterns and frequent issue types.

## Features

- **Intelligent Categorization**: Uses GPT to analyze tickets and assign business-meaningful categories
- **Similarity Detection**: TF-IDF based similarity matching ensures consistent categorization across similar tickets
- **Enterprise Ready**: Supports OAuth2 wrapper APIs for organizations that don't allow direct OpenAI access
- **Web UI**: User-friendly Streamlit interface for uploading files and viewing results
- **Excel Import/Export**: Upload ServiceNow exports and download categorized results
- **Validation**: Multiple validation layers to ensure data integrity (no data loss)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd snow-ticket-analysis-v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API credentials:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

### Option 1: Organization's GPT Wrapper API (OAuth2)

If your organization provides a wrapper API, set these in `.env`:

```env
AUTH_URL=https://your-org-auth.example.com/oauth2/token
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
GPT_WRAPPER_BASE_URL=https://your-org-gpt-wrapper.example.com
GPT_API_VERSION=2024-02-15-preview
```

The full URL is constructed as: `{base_url}/{model}/chat/completions?api-version={version}`

### Option 2: Direct OpenAI API

For direct OpenAI access:

```env
OPENAI_API_KEY=sk-your-api-key
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Configure API credentials in the sidebar

4. Upload your ServiceNow Excel export (must contain columns: Short Description, Description, Solution, Priority)

5. Click "Analyze & Categorize Tickets"

6. Download the categorized results

## Required Excel Columns

Your ServiceNow export must include:
- `Short Description`
- `Description`
- `Solution`
- `Priority`

## Project Structure

```
snow-ticket-analysis-v2/
├── app.py                  # Streamlit web UI
├── agent.py                # CLI agent (alternative to web UI)
├── config.py               # Configuration settings
├── gpt_client.py           # GPT API client (wrapper + direct)
├── categorization_engine.py # GPT-based categorization logic
├── similarity_detector.py  # TF-IDF similarity matching
├── excel_handler.py        # Excel read/write operations
├── validator.py            # Data validation
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── .streamlit/config.toml  # Streamlit configuration
```

## Security

- Never commit `.env` file (it's in `.gitignore`)
- API credentials are entered via UI or environment variables
- Application runs on localhost only by default

## License

MIT
