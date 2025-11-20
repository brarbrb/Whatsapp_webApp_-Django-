# **WhatsApp WebApp with Django**

This project is a Django-based web application that integrates machine learning pipelines for data preprocessing, fine-tuning models, and retrieval-augmented generation (RAG). It provides a platform for deploying AI-powered applications, potentially including chatbots, knowledge retrieval systems, and more.

![Alt text](Website(Django)/login.png)
---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Flow](#project-flow)
4. [Setup Instructions](#setup-instructions)
5. [Key Features](#key-features)
6. [Directory Structure](#directory-structure)
7. [Dependencies](#dependencies)

---

## **Project Overview**

Demonstration of our tool: 
[DEMO](https://drive.google.com/file/d/1t1sn073fCFUXyf07_Q2tA7RvrH61Qfek/view?usp=sharing)

This project combines:

- **Data Preprocessing**: Cleaning and transforming raw data into formats suitable for training and inference.
- **Model Fine-Tuning**: Customizing machine learning models (e.g., Tiny LLaMA) for specific tasks.
- **Retrieval-Augmented Generation (RAG)**: Implementing a pipeline that retrieves relevant knowledge and generates responses.
- **Web Application**: A Django-based interface for interacting with the models and serving processed data.

The application may also include WhatsApp integration for chatbot functionality.

**Important Note:** In our work we used Cohere and Pinecone APIs. To run this code properly one can easily recieve access on these pages: 
[Cohere Home Page](https://cohere.com/) and [Pinecone Home Page](https://www.pinecone.io/) and store them in `.env` under these names: 
```Python
COHERE_API_KEY= "your_cohere_api_key"
PINECONE_API_KEY = "your_cohere_api_key"
```
Note that both provide limited access (with daily and per minute restrictions). 

To run our distillation experiment we used the cheapest model `command-r-08-2024` (the pricing can be accessed on cohere website). The key for payed access is stored under: 
`COHERE_API_KEY_PAY`

---

## **Architecture**

The project is composed of these components:

1. **Data Preprocessing**:

   - Scripts and notebooks for cleaning and transforming raw data.
   - Converts data into formats like JSONL for training.

2. **Model Fine-Tuning**:

   - Fine-tunes a language model using preprocessed data.
   - Includes evaluation and optimization techniques like LoRA.

3. **RAG Pipeline**:

   - Combines retrieval-based methods with generative models.
   - Optimizes the pipeline using knowledge distillation.

4. **Web Application**:
   - A Django-based app for user interaction.
   - May include WhatsApp integration for communication.

---

## **Project Flow**

1. **Data Cleaning and Transformation**:

   - Raw data is processed into a structured format.
   - JSON/JSONL files are prepared for training.

2. **Model Fine-Tuning**:

   - The processed data is used to fine-tune a language model.
   - The model is evaluated for performance.

3. **RAG Pipeline**:

   - Retrieves relevant knowledge and generates responses.
   - Optimized for efficiency and accuracy.

4. **Web Application Deployment**:
   - The Django app serves as the user interface.
   - Allows interaction with the RAG pipeline and other features.

---

## **Django Setup Instructions**

### **1. Create a Project Environment**

Run the following commands in your terminal in project folder to set up a virtual environment and install Django:

```bash
cd path/to_forked_project
python3 -m venv .venv
source .venv/bin/activate # Ensure you're using the Python's enviroment
python -m pip install --upgrade pip
python -m pip install django
```

### **2. In Django Project**

Start the Django development server:

```bash
python manage.py runserver
```

To use a different port, specify it like this:

```bash
python manage.py runserver 5000
```

### **3. Runing changes**

In case of modifications to our code, changing in db. Run this lines: 
```bash
python manage.py makemigrations
python manage.py migrate
```

---

## **Key Features**

- **Data Preprocessing**: Clean and transform raw data for training.
- **Fine-Tuning**: Customize machine learning models for specific tasks.
- **RAG Pipeline**: Retrieve and generate knowledge-based responses.
- **Web Interface**: A user-friendly Django app for interaction.
- **WhatsApp Integration**: Communicate through WhatsApp (optional).

---

## **Directory Structure**

Here’s an overview of the project structure:

```
WhatsReply/
├── data_preprocessing/       # Scripts for data cleaning and transformation
├── Fine_Tune/                # Fine-tuning machine learning models
├── RAG/                      # Retrieval-Augmented Generation pipeline (We also aded small destilation technique as asmall experiment)
├── Website(Django)/          # Django-based web application
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```

---

## **Dependencies**

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```
---
