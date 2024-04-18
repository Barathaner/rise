## Setup Instructions

### Environment
- **OS:** Windows 10
- **Python Version:** 3.11

### Steps

#### 1. Clone the Repository
Clone the repository and open a terminal in the root folder of the repo.

```bash
# Example command if needed
git clone <repository-url>
cd <repository-name>
```

#### 2. Set Up Virtual Environment and Install Dependencies
Create a Python virtual environment and install all required packages from `requirements.txt`.

```bash
python -m venv venv
venv\Scripts\activate  # Activate the virtual environment on Windows
pip install -r requirements.txt
```

#### 3. Run the Application
Navigate to the source directory and start the main application.

```bash
cd src
python main.py
```

### What to Expect
After starting the application, it will take approximately 5 minutes to process on a Lenovo Legion 5. During this time, the application will:

- Execute a rule-based classifier.
- Apply the logical rules to the test split of the dataset, which constitutes 20% of the entire dataset.
- Compare the predicted classes with the actual test classes.

### Results
- **Location:** All results will be stored in the `Results` folder.
- **Contents:**
  - An interpretable presentation of each logical rule along with its coverage and accuracy.
  - The application results of these rules on the test data.
  - Overall test accuracy as a final metric.

By following these steps, you will successfully execute and analyze the rule-based classifier with detailed insights into its performance and accuracy.
