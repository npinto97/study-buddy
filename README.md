# SIIA-RS

SIIA-RS project

```plaintext
study-buddy/                   # Root del progetto
├── images/                     # Cartella per i grafici generati
│   └── pipeline_graph.png      # Grafico salvato dalla pipeline
├── pipeline/                   # Logica principale della pipeline
│   ├── __init__.py
│   ├── graph.py                # Codice che definisce il grafo e la logica della pipeline
│   └── pipeline.py             # Codice per gestire la logica della pipeline
├── config/                     # Configurazioni del progetto
│   └── config.py               # Configurazione del progetto, variabili di ambiente, etc.
├── utils/                      # Strumenti ausiliari per il progetto
│   └── graph_generator.py      # Script per generare il grafo e salvarlo in 'images'
├── tests/                      # Test del progetto
│   ├── __init__.py
│   └── test_graph.py           # Test per verificare che il grafo venga generato correttamente
├── .env                        # File per variabili di ambiente sensibili (e.g., API keys)
├── .gitignore                  # File per ignorare file/directory durante il controllo di versione
├── main.py                     # Punto di ingresso principale per eseguire la pipeline
└── README.md                   # Documentazione del progetto
└── README.md                   # Documentazione del progetto
```
