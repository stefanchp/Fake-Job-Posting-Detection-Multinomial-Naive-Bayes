import pandas as pd
import numpy as np
import re
import string
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

class MultinomialNaiveBayes:
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.priors_ = {}
        self.conditionals_ = {}
        self.classes_ = set()
        self.vocab_ = set()

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        return text.split()

    def fit(self, X, y):
        # print("Incepe antrenarea (fit)...")
        print("Starting training (fit)...")
        self.classes_ = set(y)
        word_counts_per_class = {c: defaultdict(int) for c in self.classes_}
        total_words_per_class = {c: 0 for c in self.classes_}
        doc_count_per_class = {c: 0 for c in self.classes_}
        total_docs = len(X)
        
        for text, label in zip(X, y):
            doc_count_per_class[label] += 1
            words = self._preprocess_text(text)
            for word in words:
                self.vocab_.add(word)
                word_counts_per_class[label][word] += 1
                total_words_per_class[label] += 1

        # print(f"Antrenare: Vocabular descoperit cu {len(self.vocab_)} cuvinte unice.")
        print(f"Training: Vocabulary discovered with {len(self.vocab_)} unique words.")
        
        # print("Antrenare: Se calculeaza probabilitatile Apriori P(c)...")
        print("Training: Computing prior probabilities P(c)...")
        for c in self.classes_:
            self.priors_[c] = np.log(doc_count_per_class[c] / total_docs)

        # print("Antrenare: Se calculeaza probabilitatile Conditionate P(w|c)...")
        print("Training: Computing conditional probabilities P(w|c)...")
        vocab_size = len(self.vocab_)
        
        for c in self.classes_:
            self.conditionals_[c] = {}
            total_words_in_c = total_words_per_class[c]
            for word in self.vocab_:
                word_count = word_counts_per_class[c].get(word, 0)
                numerator = word_count + self.alpha
                denominator = total_words_in_c + (self.alpha * vocab_size)
                self.conditionals_[c][word] = np.log(numerator / denominator)
        # print("Antrenarea s-a incheiat cu succes!")
        print("Training completed successfully!")

    def predict(self, X):
        predictions = []
        for text in X:
            words = self._preprocess_text(text)
            posteriors = {}
            for c in self.classes_:
                current_posterior = self.priors_[c]
                for word in words:
                    if word in self.vocab_:
                        current_posterior += self.conditionals_[c][word]
                posteriors[c] = current_posterior
            best_class = max(posteriors, key=posteriors.get)
            predictions.append(best_class)
        return predictions

STOP_WORDS = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
    'to', 'was', 'were', 'will', 'with', 'we', 'our', 'you', 'your'
])

def clean_natural_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '__NUMAR__', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    filtered_tokens = [
        word for word in tokens 
        if word not in STOP_WORDS and len(word) > 2
    ]
    return ' '.join(filtered_tokens)

def tokenize_categorical(column_name, value):
    if not isinstance(value, str):
        value = str(value)
    safe_value = re.sub(r'[\s/,-]+', '_', value)
    safe_value = re.sub(r'[^\w_]', '', safe_value)
    if safe_value == '' or safe_value.lower() == 'na':
        return f"TOKEN_{column_name.upper()}_NA"
    return f"TOKEN_{column_name.upper()}_{safe_value.upper()}"


# print("Se incarca datele...")
print("Loading data...")
try:
    data = pd.read_csv('fake_job_postings.csv')
    
    TEXT_COLS = ['title', 'location', 'department', 'salary_range', 
                 'company_profile', 'description', 'requirements', 'benefits']
    
    CAT_COLS = ['telecommuting', 'has_company_logo', 'has_questions', 
                'employment_type', 'required_experience', 
                'required_education', 'industry', 'function']
    
    TARGET_COL = 'fraudulent'
    
    # print("Se pre-proceseaza si se combina toate coloanele...")
    print("Preprocessing and combining all columns...")

    for col in TEXT_COLS:
        data[col] = data[col].fillna('')
    for col in CAT_COLS:
        data[col] = data[col].fillna('__NA__')

    all_features = []

    # print("Se curata coloanele text...")
    print("Cleaning text columns...")
    text_data_combined = data[TEXT_COLS].apply(lambda row: ' '.join(row), axis=1)
    cleaned_text = text_data_combined.apply(clean_natural_text)
    all_features.append(cleaned_text)

    # print("Se tokenizeaza coloanele cu date categoriale...")
    print("Tokenizing categorical columns...")
    for col in CAT_COLS:
        col_tokens = data[col].apply(lambda val: tokenize_categorical(col, val))
        all_features.append(col_tokens)

    combined_features_df = pd.concat(all_features, axis=1)
    
    data['full_text'] = combined_features_df.apply(lambda row: ' '.join(row), axis=1)
    
    # print(f"Am incarcat si procesat {len(data)} randuri.")
    print(f"Loaded and processed {len(data)} rows.")

except FileNotFoundError:
    # print("Eroare: Fisierul 'fake_job_postings.csv' nu a fost gasit.")
    print("Error: File 'fake_job_postings.csv' was not found.")
    data = pd.DataFrame({
        'full_text': ['test text', 'another test'], 
        'fraudulent': [0, 1]
    })


X = data['full_text'].values
y = data[TARGET_COL].values

# print("Se impart datele folosind train_test_split...")
print("Splitting data using train_test_split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25,
    random_state=42,
    stratify=y  
)

# print(f"Date impartite: {len(X_train)} esantioane de antrenare, {len(X_test)} esantioane de testare.")
print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples.")

mnb_scratch = MultinomialNaiveBayes(alpha=1.0)
mnb_scratch.fit(X_train, y_train)

# print("\n--- Evaluarea Modelului pe Setul de Test (cu sklearn.metrics) ---")
print("\n--- Model Evaluation on Test Set (with sklearn.metrics) ---")
y_pred = mnb_scratch.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# print(f"Acuratete (Accuracy): {accuracy * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")

CONFUSION_MATRIX_FILE = 'confusion_matrix.png'

# print(f"\nSe genereaza Matricea de Confuzie Avansata ({CONFUSION_MATRIX_FILE})...")
print(f"\nGenerating advanced confusion matrix ({CONFUSION_MATRIX_FILE})...")
cm = confusion_matrix(y_test, y_pred)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

annot_labels = [f"{raw_val}\n({perc_val*100:.1f}%)" 
                for raw_val, perc_val in zip(cm.flatten(), cm_norm.flatten())]
annot_labels = np.asarray(annot_labels).reshape(cm.shape)

labels = ['Legitimate (0)', 'Fraudulent (1)']

plt.figure(figsize=(10, 7))
sns.heatmap(cm_norm, 
            annot=annot_labels, 
            fmt='s',         
            cmap='Blues',      
            xticklabels=labels, 
            yticklabels=labels,
            vmin=0.0,          
            vmax=1.0,           
            cbar_kws={'label': 'Row-wise classification percentage'}) 

plt.title('Confusion Matrix (Count and Row Percentage)', fontsize=16)
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)

plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_FILE)
# print(f"Imaginea a fost salvata ca '{CONFUSION_MATRIX_FILE}'")
print(f"Image was saved as '{CONFUSION_MATRIX_FILE}'")

