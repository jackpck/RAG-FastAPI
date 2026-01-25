# RAG - FastAPI

## Introduction
This repo creates a RAG API using **FastAPI**. The original [RAG](https://github.com/jackpck/RAG) uses **streamlit** 
as the frontend. This repo aims to create a more versatile RAG endpoint for integration with any frontend choice.

## Instructions
- At the project root, run `python -m app.main`
- Enter http://localhost:8000/docs in the browser
- Under default, you will see four endpoints
    - `POST /chat`
    - `PATCH /chat-feedback`  
    - `POST /home`
    - `GET /home{input}` (dummy, not needed)
- Under `POST /chat`, click *Try it out*. Type in a query to the value of the dict and click *Execute*.
- Review the model response. Share feedback (0 or 1) by running the `PATCH /chat-feedback` endpoint. 

