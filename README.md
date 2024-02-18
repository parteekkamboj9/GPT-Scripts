# GPT-Scripts
## Project Files Overview
#### Welcome to my project repository! Here's a quick guide to the scripts you'll find:

### 1. flaskApiForSlides.py
Description:
This Flask API generates slides from text samples, utilizing MySQL for database connectivity, OpenAI models for content generation, and Langchain LLMChain for dynamic responses. It incorporates Langchain prompts for creating template objects, employs MySQLConnectionPool for efficient pooling, and integrates the Pexels.com API for fetching slide images. Please ensure to replace placeholder keys such as OpenAI and Pexels API keys before running the script. Don't forget to set up the database name (dbname).

### 2. S2T_GPT_T2S.py
Description:
This script transforms speech to text, utilizes GPT for text responses, and converts text back to speech. Leveraging the speech recognition library for audio-to-text conversion, it interacts with the OpenAI model for generating responses. Ensure you replace the OpenAI key before execution.

### 3. langchain_llms.py
Description:
Developed during my LLM learning phase, this script employs various techniques, including document loading, recursive text splitting, embedding, and vector databases such as ChromaDB, Faiss Index, Pinecone, and Deeplake. It utilizes LLM chains, retrieval prompts, and memory mechanisms like RetrievalMemory and BufferMemory. Please set up OpenAI key, Pinecone account key, environment variables, and the required indexes before running.

### 4. sampleCodePineconeLangChain.py
Description:
This script, similar to langchain_llms.py, simplifies the process using a PDF document loader, recursive text splitter, embedding, Pinecone vector database, and retrieval mechanisms. Ensure you replace the OpenAI key, Pinecone account key, environment variables, and index before executing the script.

Feel free to reach out for any clarifications or assistance. Happy coding! 🚀
