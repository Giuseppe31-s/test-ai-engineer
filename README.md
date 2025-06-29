# test-ai-engineer

## Come startare il progetto

### Requisiti
- Docker
- Docker Compose
- Python 3.11 o superiore
- [uv](https://pypi.org/project/uv/) (uv è un'alternativa a pipenv)

## Istruzioni per l'uso

### Creazione del file `.env` e configurazione
```bash
touch .env

echo "OPENAI_API_KEY=your_openai_api_key" >> .env
echo "PROVIDER=openai" >> .env
echo "QDRANT_URL=http://localhost:6333" >> .env
```

### Passaggi per avviare il progetto

1. **Creazione dell'ambiente virtuale**  
   ```bash
   uv venv
   ```

2. **Attivazione dell'ambiente virtuale**  
   - **Windows**  
     ```bash
     .\venv\Scripts\activate
     ```
   - **Linux/Mac**  
     ```bash
     source .venv/bin/activate
     ```

3. **Installazione delle dipendenze**  
   ```bash
   uv sync
   ```

4. **Avvio dei servizi Docker**  
   ```bash
   docker compose -f docker-compose.yml up
   ```

5. **Avvio del server**  
   ```bash
   python3 main.py
   ```

6. Apri il browser e vai su `http://localhost:8000` per accedere all'applicazione.

## Descrizione del progetto

Pipeline di elaborazione:

1. L’utente carica un documento in formato PDF, DOCX o TXT e identifica se è un bando o un profilo fornitore.  
2. Il documento viene estratto e convertito in testo.  
3. Il testo viene arricchito con metadati.  
4. Il testo viene trasformato in embedding denso, sparso e latente.  
5. Gli embeddings vengono salvati in un vector database.  
6. Viene calcolato uno score di similarità tra la descrizione del bando e i profili dei fornitori.  
7. Viene generata una matrice con tutti i profili e i bandi, con i rispettivi punteggi di similarità.

Non essendoci un matching diretto tra profili fornitori e bandi, è stato implementato un sistema di **re-ranking** che ordina i profili in base alla loro similarità con il bando.  
Questo approccio ibrido combina embeddings sparsi e densi per migliorare la precisione e, dopo un primo filtro, applica un modello di re-ranking per affinare ulteriormente i risultati.

## Riferimenti
- [Re-rankers con ColBERT](https://developer.ibm.com/articles/how-colbert-works/)  
- [Qdrant vector database](https://www.qdrant.tech/documentation/)  