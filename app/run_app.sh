#!/bin/bash
source venv/bin/activate

open_browser() {
  local url="http://localhost:8501"
  if command -v open >/dev/null 2>&1; then
    open "$url"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import webbrowser, time; time.sleep(2); webbrowser.open('$url')" >/dev/null 2>&1 &
  fi
}

echo "🚀 Booting Local API Gateway Interface..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &

echo "🖥️ Launching Streamlit Presentation Dashboard..."
streamlit run app/web_ui.py --server.address 0.0.0.0 --server.port 8501 &

open_browser

echo "✅ Browser launch requested. Open http://localhost:8501 manually if your browser does not open automatically."
wait
