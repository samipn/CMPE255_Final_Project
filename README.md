# Psychometric Feature Extraction for Mental Health Conversations

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

## Project Presentation

View our project presentation slides:

### Embedded Viewer
[View Presentation PDF](https://github.com/samipn/CMPE255_Final_Project/blob/main/Psychometric-Feature-Extraction-for-Mental-Health-Conversations.pdf)

### Download
[Download Presentation (PDF)](https://github.com/samipn/CMPE255_Final_Project/raw/main/Psychometric-Feature-Extraction-for-Mental-Health-Conversations.pdf)

---

## Project Overview

This project focuses on extracting psychometric features from mental health conversations to analyze and predict mental health conditions. We utilize natural language processing (NLP) techniques and machine learning models to identify patterns in text data that correlate with various mental health indicators.

## Dataset

The project uses mental health conversation datasets that include:
- Therapy session transcripts
- Online mental health support forum discussions
- Clinical interview data

## Features

Key psychometric features extracted include:
- Sentiment analysis
- Emotion detection
- Linguistic patterns
- Topic modeling
- Psychological markers

## Models

We implement and compare several machine learning models:
- Traditional ML: Logistic Regression, Random Forest, SVM
- Deep Learning: LSTM, BERT-based transformers
- Ensemble methods

## Results

Detailed results and performance metrics are available in the project notebooks and presentation.

## Installation

```bash
# Clone the repository
git clone https://github.com/samipn/CMPE255_Final_Project.git

# Navigate to project directory
cd CMPE255_Final_Project

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example usage
python main.py --input data/conversations.csv --output results/
```

## Project Structure

```
CMPE255_Final_Project/
│
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
├── models/                # Trained models
├── results/               # Output results
└── README.md             # Project documentation
```

## Contributors

This project is developed as part of CMPE 255 - Data Mining course.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
