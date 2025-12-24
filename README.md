# Tvara Interview Tasks

This repository contains the implementations for the interview tasks assigned, including DSA, API integration, and vectorization using Hugging Face.

---

## Table of Contents

- [Tasks](#tasks)  
  - [Task A – DSA](#task-a---dsa)  
  - [Task B – API Integration](#task-b---api-integration)  
  - [Task C – Vectorization](#task-c---vectorization)  
- [Setup Instructions](#setup-instructions)  
- [Usage](#usage)  
- [Notes](#notes)

---

## Tasks

### Task A – DSA
- **Problem:** [LeetCode 142: Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)  
- **Implementation:** `task_a.py`  
- **Details:**  
  - Detects the start of a cycle in a linked list.
  - Well-commented with logic explanation.
  - Uses Floyd’s Tortoise and Hare algorithm.

### Task B – API Integration
- **API:** Gemini 2.0 Flash  
- **Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`  
- **Implementation:** `task_b.py` or `gemini_cli.py`  
- **Details:**  
  - Minimal CLI interface accepting a prompt.
  - Returns generated content neatly.
  - Debug flag returns raw payload.
  - API key is handled via environment variable `.env` (not committed).

### Task C – Vectorization
- **Model:** `intfloat/e5-small-v2` (Hugging Face)  
- **Implementation:** `task_c_vectorization.py`  
- **Details:**  
  - Extracts text from a sample PDF using `pdfplumber`.  
  - Embeds sentences and performs similarity search for a query string.  
  - Outputs most relevant sentence with similarity score.

---

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Dhanush0724/Tvara-Interview-Questions.git
cd Tvara-Interview-Questions
