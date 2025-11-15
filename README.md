# SMS Spam Detection
![sms](/image.jpg)

This project implements an SMS spam detection system using machine learning techniques. The core logic and experiments are documented in `model.ipynb`.

## Overview

The goal is to classify SMS messages as either "spam" or "ham" (not spam). The workflow includes data loading, preprocessing, feature extraction, model training, evaluation, and prediction.

## Contents
#### 1. Import Libraries
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
- yfinance

#### 1. Data Loading

- The dataset is loaded (typically a CSV file with columns like `label` and `message`).
- Labels are mapped to binary values (e.g., "spam" = 1, "ham" = 0).

#### 3. Data Preprocessing

- Text cleaning: Lowercasing, removing punctuation, and stopwords.
- Tokenization: Splitting messages into words.
- Optional: Stemming or lemmatization to reduce words to their base forms.

#### 4. Feature Extraction

- Text data is converted into numerical features using techniques like:
    - Bag of Words (CountVectorizer)
    - TF-IDF (TfidfVectorizer)

#### 5. Model Training

- The dataset is split into training and testing sets.
- Machine learning models are trained, such as:
    - Naive Bayes (MultinomialNB)
    - Logistic Regression
    - Support Vector Machine (SVM)
- Hyperparameters may be tuned for better performance.

#### 6. Evaluation

- Models are evaluated using metrics like:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion matrix

#### 7. Prediction

- The trained model predicts whether new messages are spam or ham.

## Process
![Screenshot 244](/Screenshot%20(244).png)
![Screenshot 245](/Screenshot%20(245).png)
![Screenshot 246](/Screenshot%20(246).png)
![Screenshot 247](/Screenshot%20(247).png)
![Screenshot 248](/Screenshot%20(248).png)



## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/sms-spam.git
cd sms-spam
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Results

- The notebook reports model performance and may include visualizations (e.g., confusion matrix).
- The best-performing model is highlighted.

## Tools and Dependencies
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
- Environment
    - Jupyter Notebook
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```
## Project Structure
```
sms-spam/
│
├── model.ipynb  
|── model.py    
|── spam.csv  
├── requirements.txt 
├── LICENSE
├── iamge.jpg    
├── Screenshot (244).png
├── Screenshot (245).png
├── Screenshot (246).png
├── Screenshot (247).png
├── Screenshot (248).png  
└── README.md          
```
## Contributing
Contributions are welcome! If you’d like to suggest improvements — e.g., new modelling algorithms, additional feature engineering, or better documentation — please open an Issue or submit a Pull Request.
Please ensure your additions are accompanied by clear documentation and, where relevant, updated evaluation results.

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.
