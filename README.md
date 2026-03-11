# Customer_Churn_Prediction


ChurnBot
How to Run & Use the Chatbot
Step-by-Step Guide for the Marketing Team
⚠️ Before You Begin
You will need a Gemini API key to run this chatbot. Make sure you have it ready before starting. See Section 2 for details on how to add it.

1. Prerequisites
Before running the chatbot for the first time, make sure you have the following installed on your computer:

Requirement	Purpose	Download
Python 3.9 or higher	Required to run the application	python.org
pip (comes with Python)	Package manager for installing libraries	Included with Python
A modern web browser	To interact with the chat interface	Chrome / Firefox / Edge
Gemini API Key	Required to power the AI chatbot	aistudio.google.com

2. Setting Up Your Gemini API Key
🔑 IMPORTANT
The chatbot will not function without a valid Gemini API key. You must add your own API key to the config.json file before running the application for the first time.

How to Get a Free Gemini API Key
1.Go to Google AI Studio at: https://aistudio.google.com
2.Sign in with your Google account
3.Click 'Get API key' in the left sidebar
4.Click 'Create API key' and copy the key that is generated

How to Add Your API Key
5.Open the project folder on your computer
6.Find the file named config.json and open it with any text editor (Notepad, VS Code, etc.)
7.You will see this content:

{
    "GEMINI_API_KEY": "YOUR_API_KEY_HERE",
    "GEMINI_MODEL": "gemini-2.5-flash"
}

8.Replace YOUR_API_KEY_HERE with your actual Gemini API key. Example:

{
    "GEMINI_API_KEY": "AIzaSyAbc123XYZ_your_real_key_goes_here",
    "GEMINI_MODEL": "gemini-2.5-flash"
}

9.Save the file

⚠️ Keep Your Key Safe
Never share your config.json file or API key publicly. Your API key is linked to your Google account and may have usage limits or billing implications.

3. Installation & Setup
Follow these steps once to set up the project environment. You only need to do this the first time.

Step 1 — Open a Terminal
•Windows: Press Win + R, type cmd, and press Enter. Or right-click in the project folder and select 'Open in Terminal'
•Mac / Linux: Open the Terminal application

Step 2 — Navigate to the Project Folder
Use the cd command to navigate to the folder where you saved the project. For example:
cd C:\Users\YourName\churnbot-project   # Windows
cd /home/yourname/churnbot-project        # Mac / Linux

Step 3 — Create a Virtual Environment
A virtual environment keeps the project's dependencies isolated from your system Python. Run this command:
python -m venv venv

Step 4 — Activate the Virtual Environment
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
After activation, you will see (venv) at the start of your terminal prompt. This confirms the virtual environment is active.

Step 5 — Install Required Libraries
Install all the required Python packages from the requirements file:
pip install -r requirements.txt
This may take a few minutes depending on your internet connection. You will see packages being downloaded and installed.

✅ One-Time Setup Complete
You only need to complete Steps 3-5 once. After the initial setup, you can start the chatbot by simply activating the virtual environment (Step 4) and running the server (Section 4).

4. Running the Chatbot
Every Time You Want to Use the Chatbot:

10.Open your terminal and navigate to the project folder
11.Step: Activate the virtual environment:
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate

12.Step: Start the server with this command:

uvicorn main:app --reload --port 8000

You should see output similar to this in your terminal — this means the server is running successfully:
INFO:     Will watch for changes in these directories: ['...']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXXX] using WatchFiles
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
✅  ML model & encoders loaded.
INFO:     Application startup complete.

13.Open your web browser and go to:

http://127.0.0.1:8000

14.The ChurnBot chat interface will load in your browser. You are now ready to use the chatbot!

ℹ️ Stopping the Server
To stop the chatbot server, go back to your terminal window and press CTRL + C. The server will shut down gracefully.

5. Using the Chatbot — Step-by-Step
5.1 Starting a Conversation
15.When the chat interface loads, ChurnBot will greet you automatically
16.It will explain that it needs to collect approximately 19 customer details
17.Simply type your responses naturally — ChurnBot will guide you through the entire process

5.2 What Information Will ChurnBot Ask For?
ChurnBot will ask you for the following customer information one question at a time:

Field	Valid Answers / Format
Gender	Male  or  Female
Senior Citizen	Yes / No  (or  1 / 0)
Is Married	Yes  or  No
Dependents (children)	Yes  or  No
Tenure	Number of months (e.g., 24)
Phone Service	Yes  or  No
Dual (multi-line)	Yes / No / No phone service
Internet Service	DSL / Fiber optic / No
Online Security	Yes / No / No internet service
Online Backup	Yes / No / No internet service
Device Protection	Yes / No / No internet service
Tech Support	Yes / No / No internet service
Streaming TV	Yes / No / No internet service
Streaming Movies	Yes / No / No internet service
Contract Type	Month-to-month / One year / Two year
Paperless Billing	Yes  or  No
Payment Method	Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)
Monthly Charges	Dollar amount (e.g., 65.50)
Total Charges	Dollar amount (e.g., 1570.00)

5.3 Understanding the Prediction Result
Once ChurnBot has collected all the information, it will automatically run the prediction and display a result card with:

•Prediction Verdict: A clear WILL CHURN or WILL NOT CHURN label
•Churn Probability: A percentage showing how likely this customer is to churn
•Business Explanation: 4-6 bullet points explaining the key factors behind the prediction in plain business language
•Customer Profile Summary: A table showing all the collected information for your reference

💡 Example Result Interpretation
If ChurnBot shows a 78% churn probability with a WILL CHURN verdict, it means the model is highly confident this customer is at risk of leaving. The explanation bullets will tell you exactly which factors (e.g., month-to-month contract, high monthly charges, no tech support) are driving this risk.

5.4 Starting a New Prediction
To analyse a different customer, click the 'New Conversation' button in the left sidebar. This will start a fresh conversation session.

6. Troubleshooting
Common Issues & Solutions

Issue	Solution
Page not loading at http://127.0.0.1:8000	Make sure the uvicorn command is still running in your terminal. Check the terminal for error messages.
'ModuleNotFoundError' when starting server	Make sure the virtual environment is activated (you should see '(venv)' in the prompt) and that you ran pip install -r requirements.txt
'Invalid API key' or 502 error	Open config.json and verify your Gemini API key is correct with no extra spaces or quotes
ChurnBot is not responding to messages	Check your internet connection — Gemini requires an active internet connection. Also verify the API key is valid.
Server crashes with 'Address already in use'	Another process is using port 8000. Either stop that process or run: uvicorn main:app --reload --port 8001 and access http://127.0.0.1:8001
'customer_churn_model.pkl not found'	Make sure you are running the uvicorn command from inside the project folder, not from a different directory

7. Quick-Start Checklist
Use this checklist every time you want to run the chatbot:

☐	config.json has my Gemini API key
☐	Terminal is open in the project folder
☐	Virtual environment is activated — I can see (venv) in the prompt
☐	pip install -r requirements.txt was run at least once
☐	Run: uvicorn main:app --reload --port 8000
☐	Browser is open at: http://127.0.0.1:8000
☐	ChurnBot has greeted me and I am ready to start


— Happy Churn Hunting!  —
