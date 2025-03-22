# Regex Generation System
## Overview  
This project automates the generation of **regular expressions (regex)** by analyzing input data patterns. Using **deep learning (LSTM), NLP, and regex pattern matching**, it dynamically generates and refines regex patterns for structured data validation.  

## Features  
- **AI-Powered Regex Generation** – Uses LSTM-based deep learning to detect common patterns in input data.  
- **Automated Data Processing** – Reads structured data from CSV/Excel files and extracts meaningful patterns.  
- **Regex Refinement** – Continuously improves regex efficiency using a trained model on existing regex-data pairs.  
- **Pattern Type Detection** – Identifies common formats like emails, phone numbers, dates, and transactions.  
- **Customizable Training** – Allows training on new datasets for improving regex accuracy.  

## Installation  
1. **Clone the repository**  
   ```sh
   git clone https://github.com/your-username/Regex-Generator.git
2. **Navigate to the project directory**
   ```sh
   cd Regex-Generator
3. **Create a virtual environment (optional but recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
4. **Install dependencies**
   ```sh
   pip install -r requirements.txt
5. **Run the script**
   ```sh
   python Regex_generator.py

## Usage:
- Provide a CSV or Excel file containing structured data.
- The system analyzes the data and generates a regex pattern based on majority occurrences.
- The model suggests the best regex pattern for validation.
- Users can fine-tune the regex based on real-time feedback.

## Technologies Used:
- Python – Core programming language
- TensorFlow/Keras – LSTM-based deep learning model
- NLP (spaCy) – Used for analyzing text patterns
- Regex (re module) – For pattern matching and validation
- Pandas – Data handling and processing
- NumPy – Efficient numerical operations

## Contributing:
If you'd like to contribute, feel free to fork the repository, create a feature branch, and submit a pull request.

## License:
This project is open-source under the MIT License.
