# pdf-chat
Ein Python Beispiel, das dir ermöglicht OpenAI (ChatGPT) für deine eigenen PDF-Dateien zu nutzen

## Vorbereitungen

Erst musst du die notwendigen Bibliotheken installieren:

`pip install langchain openai python-dotenv faiss-cpu pypdf2 streamlit tiktoken`

Erstele eine Kopie von *.env.template* als *.env* und füge deinen OpenAI API Key ein. Die *.env* Datei wird nicht in Git eingecheckt.

## Starten der Anwendung
Da wir Streamlit nutzen, können wir nicht direkt den Python-Befehl nutzen. Statt dessen musst du folgendes Kommando ausführen:

`streamlit run pdf-chat.py`

### Starten der Anwendung in GitHub Codespaces
Man muss dafür die Sicherheitsfunktionen CORS und XSRF-Protection ausschalten:
`streamlit run pdf-chat.py --server.enableCORS false --server.enableXsrfProtection false`

Zustäzlich muss im PORTS Tab der Port 8501 auf "Public" gestellt werden.
