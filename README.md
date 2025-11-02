# Structural Analysis Visualizer

A web application for analyzing and visualizing structural test data from Excel files. Built with FastAPI and Plotly.

## Features

- 📊 **Interactive Graphs**: Generate 6 different visualization types
  - Reference case hysteresis curve
  - Test data hysteresis curve
  - Comparison plot
  - Loading history (displacement vs time)
  - Force history
  - Cumulative energy dissipation

- 📈 **Statistical Analysis**: Automatic calculation of key metrics
  - Max/min displacement and forces
  - Data point counts
  - Comparison between reference and test data

- 🎨 **Modern UI**: Clean, responsive interface with drag-and-drop file upload

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Upload your Excel file and view the generated graphs

## Excel File Format

The application expects an Excel file with the following structure:
- **Columns 0-1**: Reference case data (U in mm, F in kN)
- **Columns 4-5**: Test data (u in mm, RF in kN)
- First row contains headers
- Data starts from row 2

## API Endpoints

- `GET /`: Main application page
- `POST /upload`: Upload Excel file and receive analysis results

## Development

To run in development mode with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Technologies Used

- **Backend**: FastAPI, Python
- **Data Processing**: Pandas, Openpyxl
- **Visualization**: Plotly
- **Frontend**: HTML, CSS, JavaScript
