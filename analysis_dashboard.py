
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Configurazione Pagina
st.set_page_config(page_title="Study Buddy Analytics", page_icon="📊", layout="wide")

# Percorso File Dati
DATA_FILE = Path("data/study_results/results.csv")

def load_data():
    if not DATA_FILE.exists():
        return None
    
    try:
        import io
        
        # Expected columns for the current schema (22 columns)
        COLUMN_NAMES = [
            'timestamp', 'session_id', 
            'age', 'gender', 'enrollment',
            'sus_q1', 'sus_q2', 'sus_q3', 'sus_q4', 'sus_q5', 'sus_q6', 'sus_q7', 'sus_q8', 'sus_q9', 'sus_q10',
            'qual_completeness', 'qual_clarity', 'qual_utility', 'qual_trust', 'qual_sources',
            'nps_score',
            'comments'
        ]
        
        # Manually read and filter lines to handle mixed schemas
        valid_lines = []
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # Basic check: New schema has 22 columns -> 21 commas
                # We relax slightly to > 15 to catch minor malformations but exclude old 8-col data
                if line.count(',') >= 15:
                    valid_lines.append(line)
        
        if not valid_lines:
            st.warning("Nessun dato compatibile con il nuovo formato trovato nel CSV.")
            return None

        # Create a buffer from valid lines
        csv_buffer = io.StringIO('\n'.join(valid_lines))
        
        # Load into Pandas
        df = pd.read_csv(
            csv_buffer, 
            header=None, # We provide names explicitly, and we filtered out the old header likely
            names=COLUMN_NAMES,
            engine='python'
        )
        
        # Ensure numeric columns are numeric
        numeric_cols = [
            'sus_q1', 'sus_q2', 'sus_q3', 'sus_q4', 'sus_q5', 
            'sus_q6', 'sus_q7', 'sus_q8', 'sus_q9', 'sus_q10',
            'qual_completeness', 'qual_clarity', 'qual_utility', 'qual_trust', 'qual_sources',
            'nps_score', 'age'
        ]
        
        # Coerce to numeric, turning errors to NaN
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out rows that might be headers repeated (if any) by checking if 'age' is NaN?
        # Not strictly needed if comma count works, but good practice.
        
        st.success(f"Caricamento completato. Righe valide: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati: {e}")
        return None

def calculate_sus(row):
    """
    Calcola il System Usability Scale (SUS) score per riga.
    SUS Score = (Sum of contributions) * 2.5
    Odd items (1,3,5,7,9): Score = Response - 1
    Even items (2,4,6,8,10): Score = 5 - Response
    """
    try:
        # Check if all columns exist and are not null
        cols = [f'sus_q{i}' for i in range(1, 11)]
        if not all(c in row.index for c in cols) or row[cols].isnull().any():
            return None
        
        odd_sum = sum([row[f'sus_q{i}'] - 1 for i in [1, 3, 5, 7, 9]])
        even_sum = sum([5 - row[f'sus_q{i}'] for i in [2, 4, 6, 8, 10]])
        
        return (odd_sum + even_sum) * 2.5
    except:
        return None

def main():
    st.title("📊 Study Buddy Analytics Dashboard")
    st.markdown("Analisi in tempo reale dei risultati dello User Study.")
    
    df = load_data()
    
    if df is None or df.empty:
        st.warning("⚠️ Nessun dato trovato in `data/study_results/results.csv`.")
        return

    # --- DATA PREPROCESSING ---
    # Calculate SUS if not present or recalculate to be safe
    df['sus_score'] = df.apply(calculate_sus, axis=1)
    
    # --- KPI SECTION ---
    st.header("📈 Key Performance Indicators (KPI)")
    col1, col2, col3, col4 = st.columns(4)
    
    total_participants = len(df)
    avg_sus = df['sus_score'].mean()
    avg_nps = df['nps_score'].mean() if 'nps_score' in df.columns else 0
    
    # NPS Classification
    promoters = len(df[df['nps_score'] >= 9]) if 'nps_score' in df.columns else 0
    detractors = len(df[df['nps_score'] <= 6]) if 'nps_score' in df.columns else 0
    nps_index = ((promoters - detractors) / total_participants * 100) if total_participants > 0 else 0

    col1.metric("Totale Partecipanti", total_participants)
    col2.metric("Media SUS Score (0-100)", f"{avg_sus:.1f}" if pd.notnull(avg_sus) else "N/A")
    col3.metric("Media NPS (0-10)", f"{avg_nps:.1f}" if pd.notnull(avg_nps) else "N/A")
    col4.metric("NPS Index", f"{nps_index:.1f}")

    st.markdown("---")

    # --- DEMOGRAPHICS ---
    st.header("👥 Demografia")
    d1, d2, d3 = st.columns(3)
    
    if 'gender' in df.columns:
        with d1:
            st.subheader("Genere")
            fig = px.pie(df, names='gender', title='Distribuzione Genere')
            st.plotly_chart(fig, use_container_width=True)
            
    if 'enrollment' in df.columns:
        with d2:
            st.subheader("Iscrizione")
            fig = px.pie(df, names='enrollment', title='Tipo Iscrizione')
            st.plotly_chart(fig, use_container_width=True)

    if 'age' in df.columns:
        with d3:
            st.subheader("Età")
            fig = px.histogram(df, x='age', nbins=10, title='Distribuzione Età', text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- QUALITY METRICS ---
    st.header("⭐ Qualità e Usabilità")
    q1, q2 = st.columns(2)
    
    with q1:
        st.subheader("Distribuzione SUS Score")
        if pd.notnull(avg_sus):
            fig = px.histogram(df, x='sus_score', nbins=10, title='SUS Score Distribution', color_discrete_sequence=['#636EFA'])
            # Add vertical line for average
            fig.add_vline(x=avg_sus, line_dash="dash", line_color="red", annotation_text="Media")
            st.plotly_chart(fig, use_container_width=True)
            
            sus_grade = "F"
            if avg_sus >= 80.3: sus_grade = "A"
            elif avg_sus >= 68: sus_grade = "C" # 68 is average
            elif avg_sus >= 51: sus_grade = "D"
            
            st.info(f"Il punteggio SUS medio di **{avg_sus:.1f}** corrisponde approssimativamente a un grado **{sus_grade}**.")

    with q2:
        st.subheader("Valutazione Qualitativa (1-5)")
        qual_cols = ['qual_completeness', 'qual_clarity', 'qual_utility', 'qual_trust', 'qual_sources']
        if all(c in df.columns for c in qual_cols):
            avg_qual = df[qual_cols].mean().reset_index()
            avg_qual.columns = ['Metrica', 'Media']
            
            # Rename for nicer display
            name_map = {
                'qual_completeness': 'Completezza',
                'qual_clarity': 'Chiarezza',
                'qual_utility': 'Utilità',
                'qual_trust': 'Fiducia',
                'qual_sources': 'Fonti'
            }
            avg_qual['Metrica'] = avg_qual['Metrica'].map(name_map)
            
            fig = px.bar(avg_qual, x='Metrica', y='Media', range_y=[1,5], title='Media Punteggi Qualitativi', text_auto='.2f', color='Media', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

    # --- RAW DATA ---
    with st.expander("📄 Visualizza Dati Grezzi"):
        st.dataframe(df)

if __name__ == "__main__":
    main()
