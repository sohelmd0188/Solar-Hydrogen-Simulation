
# Solar-Hydrogen simulation

This project contains a Streamlit-based interactive simulation and dashboard for a Solar-Hydrogen techno-economic system.
It is prepared for research and journal-ready exploratory analysis.

## Files
- `solar_h2_simulation.py` : Streamlit app (main)
- `requirements.txt` : Python package requirements
- `sample_results.csv` : Example output values

## How to run locally
1. (Optional) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run solar_h2_simulation.py
   ```
4. The app will open at http://localhost:8501

## How to run in Google Colab
You can run Streamlit in Colab using `ngrok` or `localtunnel`. Alternatively, you can copy the core calculation cells into a Colab notebook.

## Notes
- The formulas and baseline values are derived from the user's uploaded paper.
- The app includes an optional Monte-Carlo quick sampling for LCOH (2000 iterations).
