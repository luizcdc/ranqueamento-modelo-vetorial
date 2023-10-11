import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import nltk

# Uncomment the following lines to download the necessary NLTK packages if you haven't already:
# nltk.download('wordnet')
# nltk.download('punkt')

# Initialize stemmer and lemmatizer. These are used for reducing words to their root form.
stemmer = PorterStemmer()  # The Porter Stemming algorithm is a process for removing the commoner morphological and inflexional endings from words in English.
lemmatizer = WordNetLemmatizer()  # Lemmatization is the process of grouping together the inflected forms of a word so they can be analysed as a single item.

df = pd.read_csv('doc.csv') 

# Define a function to preprocess the text
def preprocess(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Tokenize the text (split it into individual words)
    tokens = word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    tokens = [token for token in tokens if token.isalnum()]
    
    # Remove English stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    # Apply stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Apply lemmatization
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]  # 'v' stands for verb
    
    # Join the tokens back into a single string and return it
    return ' '.join(tokens)

# Apply the preprocessing function to the 'Abstract' column of the DataFrame
df['Abstract'] = df['Abstract'].apply(preprocess)

# Create a TF-IDF vectorizer and use it to transform the 'Abstract' column into a matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Abstract'])


# Define a function for term weighting search
def term_weighting_search(weight1, weight2, query1, query2, and_or):

    df = pd.read_csv('doc.csv')

    # Combine the 'Title', 'Link', and 'Abstract' columns into one
    df['combined'] = df['Title'] + ' ' + df['Link'] + ' ' + df['Abstract']

    # Initialize a TF-IDF Vectorizer and fit it on combined text data
    tfidf_vectorizer2 = TfidfVectorizer()
    tfidf_vectorizer2.fit(df['combined'])

    # Transform queries and documents into TF-IDF vectors
    tfidf_query1 = tfidf_vectorizer2.transform([query1]).toarray()[0] * weight1
    tfidf_query2 = tfidf_vectorizer2.transform([query2]).toarray()[0] * weight2
    tfidf_docs = tfidf_vectorizer2.transform(df['combined'])

    if and_or == "and":
        # AND operation: minimum of the two query vectors
        tfidf_query = np.minimum(tfidf_query1, tfidf_query2)
    elif and_or == "or":
        # OR operation: maximum of the two query vectors
        tfidf_query = np.maximum(tfidf_query1, tfidf_query2)
    elif and_or == "not":
        # NOT operation: inverse of the first query vector
        tfidf_query = 1 - tfidf_query1

    # Compute cosine similarity between the query and all documents
    similarities = cosine_similarity([tfidf_query], tfidf_docs)

    # Get the indices of the documents sorted by their similarity score
    sorted_indices = np.argsort(similarities[0])[::-1]

    # Print the top 5 most relevant documents
    print("Top 5 Relevant Documents:")
    result_boolean_search = df.iloc[sorted_indices[:5]]
    if not result_boolean_search.empty:
        print(result_boolean_search)
    else:
        print("No relevant documents found.")


'''# Example usage:
weight1 = 0.1  # Change as needed
weight2 = 0.2  # Change as needed
query1 = "machine"
query2 = "learning"
and_or = "and"  # Change to "or" or "not" as needed

term_weighting_search(weight1, weight2, query1, query2, and_or)
'''

def proximity_search(query, text):
    # Use regular expressions to parse the query
    match = re.search(r'(\w+) NEAR/(\d+) (\w+)', query, re.IGNORECASE)
    
    if match:
        # Extract the two words and the distance from the query
        term1, distance, term2 = match.groups()
        distance = int(distance)

        # Find all occurrences of the terms in the text
        term1_indices = [m.start() for m in re.finditer(term1, text, re.IGNORECASE)]
        term2_indices = [m.start() for m in re.finditer(term2, text, re.IGNORECASE)]

        # Check all pairs of term1 and term2 occurrences
        for i in term1_indices:
            for j in term2_indices:
                # Calculate the number of words between term1 and term2
                between = text[min(i,j):max(i,j)].count(' ')

                # If the number of words is less than or equal to the distance, return True
                if between <= distance:
                    return True

    # If no pair of occurrences is within the distance, return False
    return False


def proximity_search_operation(first_word, second_word, distance):
    column = 'Abstract'

    df = pd.read_csv('doc.csv', encoding='utf-8')

    # Check if df and column are valid
    if df is None or column not in df.columns:
        raise ValueError("Invalid DataFrame or column name")

    # Perform proximity search on each row of the DataFrame
    relevant_docs_proximity = df[df[column].apply(lambda x: proximity_search(first_word + " NEAR/" + str(distance) + " " + second_word, x))]

    # Print relevant documents for each type of search
    print("Relevant Documents (Proximity Search):")
    if not relevant_docs_proximity.empty:
        print(relevant_docs_proximity)
    else:
        print("No relevant documents found.")



# Define our search terms and the maximum allowed distance between them
#first_word = "Academic"
#second_word = "attention"
#distance = 8

# Call the function
#proximity_search_operation(df, 'Abstract', first_word, second_word, distance)


def fuzzy_search_operation(query):
    # Set a threshold for the similarity score
    threshold = 70

    # Perform fuzzy matching on each row of the DataFrame
    relevant_docs_fuzzy = df[df['Abstract'].apply(lambda x: process.extractOne(query, x.split(), scorer=fuzz.token_sort_ratio)[1] >= threshold)]

    # Print relevant documents for each type of search
    print("Relevant Documents (Fuzzy Matching):")
    if not relevant_docs_fuzzy.empty:
        print(relevant_docs_fuzzy)
    else:
        print("No relevant documents found.")

#fuzzy_search_operation("artiifcial")
#aacfiiilrt
#aacfiiilrt

def conceptual_search_operation(query):
    # Load a pre-trained BERT model (this could take a while)
    model = SentenceTransformer('msmarco-distilbert-base-v2')

    df = pd.read_csv('doc.csv')

    # We will use the 'Abstract' column as the documents
    documents = df['Abstract'].tolist()

    # Suppose we have the following query
    #query = "pedagogy"

    # Use BERT to transform the documents and query into vectors
    document_vectors = model.encode(documents)
    query_vector = model.encode([query])

    # Compute cosine similarities between the query and all documents
    similarities = cosine_similarity(query_vector, document_vectors)

    # Get the indices of the documents sorted by their similarity score
    sorted_indices = similarities.argsort()[0][::-1]

    # Print the top 5 most relevant documents along with their titles and links
    print("Top 5 Relevant Documents:")
    for index in sorted_indices[:5]:
        print(f"Title: {df.iloc[index]['Title']}")

#conceptual_search_operation("pedagogy")

'''     
# Using tfidf and cosine similarity, but in teste.py it has a truly conceptual search
def conceptual_search(query):
    # Initialize a TF-IDF Vectorizer and fit it on our 'Abstract' data
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Abstract'])

    # Transform the query into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query])

    # Compute cosine similarity between the query and all documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort the documents by their similarity score in descending order and return the sorted DataFrame
    return df.iloc[np.argsort(cosine_similarities)[::-1]]


def conceptual_search_operation(query):
    # Perform conceptual search on the DataFrame
    relevant_docs_conceptual = conceptual_search(query)

    # Print relevant documents for each type of search
    if not relevant_docs_conceptual.empty:
        print("Relevant Documents (Conceptual Search):")
        print(relevant_docs_conceptual)
    else:
        print("No relevant documents found.")

'''
#query = "cryptography"
#relevant_docs = conceptual_search_operation(query)


# Define our DataFrame 'df' and TfidfVectorizer 'tfidf_vectorizer'
# For the purpose of this example, let's assume they are defined and trained on our corpus

'''
df = pd.read_csv('doc.csv', encoding='utf-8')

# Define and train our TfidfVectorizer 'tfidf_vectorizer'
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(df['Abstract'])

def term_boosting_operation(query, df, tfidf_vectorizer):
    # Term Boosting with real-valued weights
    boosted_query = " ".join([term if "^" not in term else term.split("^")[0] for term in query.split()])
    boosted_weights = [float(term.split("^")[1]) if "^" in term else 1 for term in query.split()]
    
    # Calculate the sum of the boosted TF-IDF vectors instead of the product
    boosted_result = np.sum([tfidf_vectorizer.transform([term]).toarray() ** weight for term, weight in zip(boosted_query.split(), boosted_weights)], axis=0)

    # Return the relevant documents instead of printing them
    if boosted_result.any():
        # Get the sorted indices
        sorted_indices = np.argsort(boosted_result[0])[::-1]
        
        # Check if the indices are within the bounds of df
        valid_indices = sorted_indices[sorted_indices < len(df)]
        
        return df.iloc[valid_indices]
    else:
        return None

# Example queries
query8 = "automatic^5"  # Term Boosting

# Call the function and store the result
result = term_boosting_operation(query8, df, tfidf_vectorizer)

# Print the result
print("Relevant Documents (Term Boosting):")
if result is not None:
    print(result)
else:
    print("No relevant documents found.")
'''

def field_based_search_operation(query, field_name):
    # Initialize a TF-IDF Vectorizer and fit it with our specified field data
    field_tfidf_vectorizer = TfidfVectorizer()
    field_tfidf_matrix = field_tfidf_vectorizer.fit_transform(df[field_name])

    # Transform the query into a TF-IDF vector
    query_vector = field_tfidf_vectorizer.transform([query])

    # Compute cosine similarity between the query and all documents
    cosine_similarities = cosine_similarity(query_vector, field_tfidf_matrix).flatten()

    # Print relevant documents for each type of search
    if cosine_similarities.any():
        print("Relevant Documents (Field-Based Search):")
        print(df.iloc[np.argsort(cosine_similarities)[::-1]])
    else:
        print("No relevant documents found.")


'''query = "Cryptographic"
query = "Liability"
relevant_docs_title = field_based_search_operation(query, "Title", df)
if relevant_docs_title is not None:
    print("Relevant Documents (Field-Based Search):")
    print(relevant_docs_title)
else:
    print("No relevant documents found.")
'''

def exclusion_search_operation(query):
    # Split the query into terms
    terms = query.split()

    # Identify the terms to be excluded (those that start with '-')
    excluded_terms = [term[1:] for term in terms if term.startswith('-')]

    # Perform the exclusion search on the 'Abstract' column of the DataFrame
    # The 'any' function returns True if any element of the iterable is true
    relevant_docs_exclusion = df[df['Abstract'].apply(lambda x: not any(term in x for term in excluded_terms))]

    # Print the result
    print("Relevant Documents (Exclusion Search):")
    if not relevant_docs_exclusion.empty:
        print(relevant_docs_exclusion)
    else:
        print("No relevant documents found.")


# Example queries
#query7 = "artificial intelligence -deep learning"  # Exclusion Search

# Call the function and store the result
#exclusion_search_operation(query7)