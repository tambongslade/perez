# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A FastAPI-based structural analysis visualizer that processes Excel files containing test data and generates interactive plots comparing reference case data with test specimens. Features AI-powered analysis reports using OpenAI's GPT models.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run production server
python main.py
```

## Architecture

**Single-file application** (`main.py`) with FastAPI backend serving:
- Static HTML frontend (`static/index.html`)
- File upload and processing endpoints
- AI report generation via OpenAI API
- PDF export functionality

**Key Components:**
- `process_excel_file()`: Extracts datasets from Excel columns 0-1 (reference) and 4-5 (test data)
- `create_plots()`: Generates 6 plot types using Plotly (hysteresis curves, comparison, loading history, force history, energy dissipation)
- `calculate_engineering_metrics()`: Computes stiffness, energy dissipation, displacement/force ranges
- `generate_ai_report()`: Creates technical analysis using GPT-4o-mini

**Data Format:** Excel files with reference case in columns 0-1 (U mm, F kN) and test data in columns 4-5 (u mm, RF kN), starting from row 2.

## Environment Setup

Requires `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## API Endpoints

- `GET /`: Serves main application page
- `POST /upload`: Processes Excel file, returns plots and statistics
- `POST /generate-ai-report`: Creates AI analysis report
- `POST /export-pdf`: Exports report as PDF

## Dependencies

Core: FastAPI, pandas, plotly, openpyxl for data processing and visualization
AI: openai for report generation
PDF: reportlab for export functionality
Frontend: Vanilla JavaScript with Plotly.js for interactive charts
