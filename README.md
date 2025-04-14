# Resume Screening Application

This project is a Resume Screening Application developed using Python. It leverages Natural Language Processing (NLP) and Machine Learning techniques to automate the process of screening resumes based on job categories. The application allows users to upload candidate resumes and predicts their suitability for specified job roles.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- **Resume Upload:** Allows users to upload candidate resumes in PDF format.
- **Resume Cleaning:** Processes and cleans resume text by removing URLs, special characters, and non-ASCII characters.
- **Job Category Selection:** Users can select the desired job category to match candidate resumes against.
- **Prediction:** Utilizes a machine learning model to predict the job category of the uploaded resume.
- **Matching:** Compares the predicted category with the selected job category to determine suitability.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository:**

```bash
   git clone https://github.com/Miheret-Girmachew/Resume-Screening.git
   cd Resume-Screening
```

** For windows**

```bash
  python -m venv venv
  venv\Scripts\activate
```

** For macOS/Linux **

```bash
  python3 -m venv venv
  source venv/bin/activate
```

** Install Dependencies **

```bash
pip install -r requirements.txt
```

** Run the Streamlit Application **

```bash
streamlit run app.py.
```

** To run the data analysis script **

```bash
python data_analysis.py
``
