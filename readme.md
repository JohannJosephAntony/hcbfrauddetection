Fraud Detection Dashboard
Project Overview
This project is a dynamic web-based dashboard designed to help identify potentially fraudulent transactions. It leverages machine learning to predict fraud probabilities and provides an interactive interface for visualizing transaction data, either globally or filtered by specific organizations.

Features
Dashboard Overview: Displays key metrics like total transactions, total fraud count, and overall fraud rate.

Daily Transaction Volume: Visualizes the sum of transaction amounts per day.

Fraud by Transaction Type: Shows the count of fraudulent transactions categorized by their type.

Latest Transactions: Presents a table of the 20 most recent transactions, including their predicted fraud probability.

Organization Filtering: Allows users to view data specific to a particular organization (e.g., 'scrapbook', 'hack-club') using an input field.

Machine Learning Integration: Utilizes a pre-trained Logistic Regression model to calculate fraud probabilities for transactions.

Synthetic Data Fallback: Can generate and use synthetic data for development and testing if a live API connection is not configured.

Technologies Used
Backend:

Python 3.x

Flask: Web framework for building the API and serving HTML.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For the machine learning model (Logistic Regression).

python-dotenv: For managing environment variables (like API keys).

Requests: For making HTTP requests to external APIs (HCB API).

Joblib: For saving and loading the trained machine learning model.

Gunicorn: A production-ready WSGI HTTP server for deploying Flask applications.

Frontend:

HTML5: Structure of the web page.

JavaScript: For fetching data from the backend API and dynamically updating the dashboard.

Tailwind CSS: For modern, responsive styling with a light blue theme.

Setup Instructions
Follow these steps to get the project up and running on your local machine.

1. Clone the Repository
git clone <your-repository-url>
cd your-fraud-dashboard

2. Set up a Virtual Environment (Recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
Install all the required Python packages:

pip install -r requirements.txt

4. Prepare Environment Variables
Create a file named .env in the root directory of your project. You can use the provided .env.example as a template.

# .env.example content:
# HCB_API_KEY="your_hack_club_bank_api_key_here"

Important: If you want to fetch real data from the Hack Club Bank API, you will need to obtain an API key and set it in your .env file. Without it, the application will automatically generate and use synthetic data. Do NOT commit your actual .env file to GitHub!

5. Generate Synthetic Data and Train the Model
These scripts will prepare the data and train the fraud detection model, saving it as fraud_detector_model.pkl.

python generate_data.py
python train_model.py

6. Run the Flask Application
Start the Flask development server:

python app.py

You should see output indicating the server is running, typically on http://127.0.0.1:5000.

Usage
Access the Dashboard: Open your web browser and navigate to http://127.0.0.1:5000.

Global View: The dashboard will initially display a "Global (Synthetic)" view, showing aggregated data from the generated synthetic transactions.

Filter by Organization: To view data for a specific organization, type its name (e.g., scrapbook or hack-club) into the "Enter Organization Name" input box and click "Load Organization Data." The dashboard will update to show data for that specific pseudo-organization from the synthetic dataset.

Screenshots
<!-- Add screenshots of your dashboard here! -->

<!-- Example: -->

<!--  -->

<!--  -->

Challenges and Learnings
Handling Missing Data (NaNs): Encountered NaN values in JSON responses and model inputs. Solved by implementing .fillna(0) at various stages of data processing and before sending data to the frontend.

Local vs. Deployment Server: Understood the difference between python app.py for local development and gunicorn for production deployment, including binding to dynamic ports.

API Integration: Learned how to fetch and process data from external APIs, and how to gracefully handle scenarios where API keys are missing or data fetching fails.

Frontend Styling with Tailwind CSS: Gained practical experience in applying a utility-first CSS framework to create a clean and responsive user interface.

Disclaimer: This dashboard is not affiliated with Hack Club.
