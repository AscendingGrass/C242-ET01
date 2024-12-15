# Dicoding Generative AI

Indonesia’s educational technology sector faces significant challenges in leveraging Generative AI, particularly with Large Language Models (LLMs) like Google's Gemini for Bahasa Indonesia on Google Cloud. This project addresses these challenges by building a Generative AI system tailored for Dicoding, enabling automated, curriculum-aligned content generation and improving contextual accuracy in forums and chats.

## C242-ET01
This repository contains code and resources for fine-tuning and testing AI models for this project, including two Colab Notebooks and a dataset.

## IMPORTANT DEPLOY LINK
Due to changes made directly by Google Cloud, we were forced to update the deployment link, which now redirects to the following link:
[Dicoding Generative AI](https://s.id/DicodingGENAI)

## Repository Structure
Our repository is organized into the following directories & files:

* `dataset`: This directory contains the dataset used in this project.
* `results`: This directory contains the test results of our models and other openly available models on our test set.
* `C242_ET01_Llama_3_1_8b_+_Unsloth_2x_faster_finetuning.ipynb`: This is a Colab Notebook that contains our fine-tuning code. The train set can be found in the `dataset` directory.
* `C242_ET01_Benchmark.ipynb`: This is a Colab Notebook for testing our models. The results can be found in the `results` directory and the test set can be found in the `dataset` directory.
* `self-hosted-ai-starter-kit`: This is a custom docker file from [self-hosted-ai-starter-kit](https://github.com/n8n-io/self-hosted-ai-starter-kit). For installation you can follow the instruction inside the link or you can follow this tutorial [Run ALL Your AI Locally in Minutes (LLMs, RAG, and more)](https://www.youtube.com/watch?v=V_0dNE-H2gw&t=640s)

## Models
Our finetuned models are available here:
* [Llama-3.2_3B_C242-ET01](https://huggingface.co/AscendingGrass/Llama-3.2_3B_C242-ET01)
* [Llama-3.1_8B_C242-ET01](https://huggingface.co/AscendingGrass/Llama-3.1_8B_C242-ET01)

The 4bit GGUF version of those models are available here:
* [Llama-3.2_3B_C242-ET01_GGUF](https://huggingface.co/AscendingGrass/Llama-3.2_3B_C242-ET01_GGUF)
* [Llama-3.1_8B_C242-ET01_GGUF](https://huggingface.co/AscendingGrass/Llama-3.1_8B_C242-ET01_GGUF)

## Webapps
Our webapps for better visualization and user-friendly:
* [For this Project](https://github.com/RayaSatriatama/Dicoding-GenAI-WebApps/tree/N8N-Implementation-with-CRUD-RAG)
* [Without RAG and Gemini Only](https://github.com/RayaSatriatama/Dicoding-GenAI-WebApps)
