# WebChat

WebChat is an innovative application that allows you to create a chatbot for any website using Retrieval-Augmented Generation (RAG) combined with a Large Language Model (LLM). The chatbot can answer questions based on the content of the provided website.

## Table of Content
- Introduction
- Features
- Requirements
- Installation
- Demo

## Introduction

WebChat leverages advanced techniques in natural language processing to create an intelligent chatbot capable of interacting with website content. By utilizing RAG and LLM, WebChat provides accurate and contextually relevant responses to user queries based on the information available on the website.

## Features

- Content Crawling: Fetches URLs from the specified website and extracts all URLs having given URLs as base URL.
 - Dynamic URL Exploration: Allows users to define the range (1 to 100) of URLs to be explored for creating the knowledge base.
- Intelligent Chatbot: Uses RAG and LLM to generate responses based on website content.
- Scalable Vector Embedding: Loads vector embeddings to efficiently search and retrieve information from the website.
- Interactive Interface: User-friendly interface to interact with the chatbot and ask questions.

## Installation
```bash
git clone https://github.com/alphaace001/WebChat
cd WebChat
pip install -r requirements.txt
streamlit run project.py
```

## Requirements 
- Ollama
- Python

- ## Demo
