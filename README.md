# Medicine Recommendation System

A web-based application that provides personalized medical recommendations based on symptom analysis using machine learning.

## Overview

This project is a Streamlit-based web application that helps users:
- Predict potential diseases based on their symptoms
- Get personalized medication recommendations
- Receive diet suggestions
- Get exercise/workout recommendations
- View disease descriptions and precautions

## Features

- Disease prediction using a trained neural network model
- Personalized medication recommendations
- Diet suggestions based on disease
- Exercise/workout recommendations
- Detailed disease information and precautions
- Interactive user interface with Streamlit
- Real-time symptom analysis

## Tech Stack

- **Backend**: Python 3.x
- **Framework**: Streamlit
- **Machine Learning**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Data Storage**: CSV files

## Project Structure

```
project/
├── app.py                 # Main application file
├── data/                 # Dataset files
├── models/              # ML model and label encoder
├── notebooks/           # Jupyter notebooks
├── requirements.txt     # Python dependencies
└── pyproject.toml      # Project configuration
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```bash
   # Windows
   .\.venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure all dependencies are installed
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in your web browser

## Data Sources

The project uses several datasets:
- Training data for disease prediction
- Symptom severity data
- Medication information
- Diet recommendations
- Workout suggestions
- Disease descriptions and precautions

## Model Architecture

The project uses a neural network model trained on medical symptom data to predict diseases based on user input symptoms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical datasets used in this project
- Streamlit framework
- TensorFlow/Keras for machine learning implementation