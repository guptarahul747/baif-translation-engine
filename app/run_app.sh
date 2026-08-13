#!/bin/bash
source venv/bin/activate

echo "🚀 Booting Local API Gateway Interface..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &

echo "🖥️ Launching Streamlit Presentation Dashboard..."
streamlit run app/web_ui.py --server.port 8501
