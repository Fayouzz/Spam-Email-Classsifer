# Spam Email Classifier (under development)

## Overview
My first ML project. This very basic project is a spam email classifier that uses ML techniques to identify whether an email is spam or not. It utilizes a dataset of labeled emails to train and evaluate the model's performance.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Technologies and Frameworks](#technologies-and-frameworks)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Project Description
This spam email classifier is designed to distinguish between spam and non-spam (ham) emails. It's built using machine learning algorithms and leverages the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to convert email text data into numerical features.

### Key Features
- Preprocessing of email text data, including tokenization and TF-IDF vectorization.
- Training of a machine learning model (e.g., Multinomial Naive Bayes) on a labeled dataset.
- Evaluation of model performance using metrics such as accuracy, precision, recall, and F1-score.

## Dataset
The dataset used for this project contains a collection of emails labeled as 'spam' and 'ham'. The dataset has been dedicated to the public domain, allowing free use, modification, distribution, and even commercial use without requiring permission. The dataset is included in this repository as [spam.csv](spam.csv).

### Dataset Source
The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email) and is available under a public domain dedication.

## Technologies and Frameworks
- Python.
- Scikit-Learn: For ML and model evaluation.
- Pandas: For data loading and preprocessing.
- Git and GitHub: Version control and repository hosting.
- Jupyter Notebook (Optional): For experimentation and documentation.

## Usage
1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Fayouzz/Spam-Email-Classsifer.git
2. Navigate to the project directory:
   ```bash
   cd spamemailclassifier
3. Install the required Python packages:
   ```bash
   pip install scikit-learn pandas
4. Run the spam email classifier:
   ```bash
   python spamemailclassifier.py
Be sure to replace spamemailclassifier.py with the actual filename of your Python script.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Fayouzz/Spam-Email-Classsifer/blob/main/LICENSE) file for details.

## Contact
Any suggestions, feedback or questions about this project would be appreciated. Feel free to reach out to me at [Fayouz](mailto:chappufayouz@gmail.com).
