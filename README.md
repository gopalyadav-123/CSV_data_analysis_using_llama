# CSV Data Analysis with Llama2

Welcome to CSV Data Analysis with Llama2! This application allows you to upload a CSV file and perform various operations such as chatting with your data using natural language, performing statistical analysis, and generating different types of plots.

## Features

- **Chat with CSV**: Interact with your data using natural language queries.
- **Statistical Analysis**: Get descriptive statistics of your data.
- **Generate Plots**: Visualize your data using different plot types.

## How to Use

1. **Upload a CSV file** using the sidebar.
2. Navigate to:
   - **Chat with CSV**: Interact with your data using natural language queries.
   - **Statistical Analysis**: Get descriptive statistics of your data.
   - **Generate Plots**: Visualize your data using different plot types.
3. Start exploring and analyzing your data effortlessly!

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gopalyadav-123/CSV_data_analysis_using_llama.git
   cd CSV_data_analysis_using_llama
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Run the application**: 
   ```bash
   streamlit run app.py
   
## Dependencies
The application relies on the following main packages:

- **streamlit**: Web framework for building interactive applications.
- **pandas**: Data manipulation and analysis library.
- **seaborn and matplotlib**: Libraries for creating plots and visualizations.
- **langchain_community and langchain_huggingface**: Libraries for LLM-based data interactions.
- **faiss**: Library for efficient similarity search and clustering of dense vectors.
## Example
After running the application, you will see a welcome screen. Upload your CSV file using the sidebar and navigate through the available options to interact with and analyze your data.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact
For any questions or inquiries, please contact gopalyadav80570@gmail.com.
