import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def cleaned_text(txt):
    # Function for removing the punctuation marks
    def remove_punct(txt):
        txt_nopunct = "".join([c for c in txt if c not in string.punctuation]).lower()
        return txt_nopunct

    # Function for removing links
    def remove_links(txt):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        clean_text = url_pattern.sub('', txt)
        return clean_text

    # Function for removing non-English characters and special characters
    def remove_non_english_special_chars(txt):
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
        return cleaned_text

    # Function for tokenizing and removing stopwords
    def tokenize_remove_sw(txt):
        txt_tokenized = re.split('\W', txt)
        txt_no_sw = [word for word in txt_tokenized if word.lower() not in stopwords]
        return txt_no_sw

    # Function for lemmatizing words
    def lemmatization(txt):
        text_lemmatized = [lemmatizer.lemmatize(word) for word in txt]
        return ' '.join(text_lemmatized)

    # Function for removing extra spaces
    def remove_extra_spaces(txt):
        return " ".join(txt.split())

  
    cleaned = remove_punct(txt)
    cleaned = remove_links(cleaned)
    cleaned = remove_non_english_special_chars(cleaned)
    cleaned = tokenize_remove_sw(cleaned)
    cleaned = lemmatization(cleaned)  
    cleaned = remove_extra_spaces(cleaned) 

    return cleaned 

from keybert import KeyBERT
kw_model = KeyBERT()

# KeyBert function that extracts the keywords
def get_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2),stop_words="english")
    keywords = " ".join([k[0] for k in keywords])
    return keywords

# Function to update DataFrame and calculate TF-IDF features and cosine similarity
def update_dataframe_and_calculate_similarity(user_input, model_df):
    cleaned_result = cleaned_text(user_input)
    keywords = get_keywords(cleaned_result)

    # Add a new row with cleaned input and keywords
    model_df.loc[len(model_df.index)] = [cleaned_result, "", keywords]

    # Calculate TF-IDF features
    tfidf = TfidfVectorizer(analyzer='word', min_df=3, max_df=0.6, stop_words="english", encoding='utf-8', token_pattern=r"(?u)\S\S+")
    tfidf_encoding = tfidf.fit_transform(model_df["Keyword"])

    # Calculate cosine similarity
    prog_cosine_sim = cosine_similarity(tfidf_encoding, tfidf_encoding)

    return cleaned_result, keywords, prog_cosine_sim

# [FOR PAPER] Function to update DataFrame and calculate TF-IDF features and cosine similarity
def paper_update_dataframe_and_calculate_similarity(user_input, paper_df):
    cleaned_result = cleaned_text(user_input)
    keywords = get_keywords(cleaned_result)

    # Add a new row with cleaned input and keywords
    paper_df.loc[len(paper_df.index)] = [cleaned_result, "", "", keywords]

    # Calculate TF-IDF features
    tfidf = TfidfVectorizer(analyzer='word', min_df=3, max_df=0.6, stop_words="english", encoding='utf-8', token_pattern=r"(?u)\S\S+")
    tfidf_encoding = tfidf.fit_transform(paper_df["Keyword"])

    # Calculate cosine similarity
    prog_cosine_sim = cosine_similarity(tfidf_encoding, tfidf_encoding)

    return cleaned_result, keywords, prog_cosine_sim

# Function to recommend programming languages
def recommend_prog_lang_similar_to(prog_name, threshold=0.2, n=5, cosine_sim_mat=None, programs_list=None, projects=None):
    if cosine_sim_mat is not None and programs_list is not None and projects is not None:
        # get index of the input programming language
        input_idx = programs_list[programs_list == prog_name].index[0]

        # Find top n similar programming languages with decreasing order of similarity score
        top_n_prog_idx = list(pd.Series(cosine_sim_mat[input_idx]).sort_values(ascending=False).iloc[1:n+1].index)

        # extract similarity scores
        similarity_scores = pd.Series(cosine_sim_mat[input_idx]).sort_values(ascending=False).iloc[1:n+1].values

        # display only recommendations with similarity scores >= threshold
        recommended_prog = [[programs_list[i], projects[i], '{:.2f}'.format(similarity_scores[j]*100)] for j, i in enumerate(top_n_prog_idx) if similarity_scores[j] >= threshold]

        # recommended project list
        # eliminate duplicate recommendations and only store recommendation with the highest percentage
        reco = []
        program_set = set()
        for program, project, percentage in recommended_prog:
            if program not in program_set:
                reco.append([program, percentage])
                program_set.add(program)
            else:
                for item in reco:
                    if item[0] == program and float(item[1].replace('%', '')) < float(percentage):
                        item[1] = percentage

        project_set = [[project, percentage] for program, project, percentage in recommended_prog]
        return reco, project_set

    return None, None

# [PAPER] Function to recommend programming languages
def paper_recommender(paper_name, threshold=0.2, n=10, cosine_sim_mat=None, papers_list=None, link_list=None, authors_list=None):
    if cosine_sim_mat is not None and papers_list is not None and link_list is not None and authors_list is not None:
        # get index of the input programming language
        input_idx = papers_list[papers_list == paper_name].index[0]
        # Find top n similar programming languages with decreasing order of similarity score
        top_n_prog_idx = list(pd.Series(cosine_sim_mat[input_idx]).sort_values(ascending=False).iloc[1:n+1].index)

        # extract similarity scores
        similarity_scores = pd.Series(cosine_sim_mat[input_idx]).sort_values(ascending=False).iloc[1:n+1].values

        # display only recommendations with similarity scores >= threshold
        recommended_paper = [[papers_list[i], link_list[i], authors_list[i], '{:.2f}'.format(similarity_scores[j]*100)] for j, i in enumerate(top_n_prog_idx) if similarity_scores[j] >= threshold]

        #recommended project list

        return recommended_paper

    return None

# Function for sorting lists using merge sort
def merge_sort(arr, descriptions):
    if len(arr) <= 1:
        return arr, descriptions

    mid = len(arr) // 2
    left_half, left_descriptions = merge_sort(arr[:mid], descriptions[:mid])
    right_half, right_descriptions = merge_sort(arr[mid:], descriptions[mid:])

    result = []
    result_descriptions = []
    i = j = 0

    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_half[j]:
            result.append(left_half[i])
            result_descriptions.append(left_descriptions[i])
            i += 1
        else:
            result.append(right_half[j])
            result_descriptions.append(right_descriptions[j])
            j += 1

    result.extend(left_half[i:])
    result_descriptions.extend(left_descriptions[i:])
    result.extend(right_half[j:])
    result_descriptions.extend(right_descriptions[j:])

    return result, result_descriptions

# Function for binary search
def binary_search(sorted_list, descriptions, target):
    low, high = 0, len(sorted_list) - 1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] == target:
            return mid, descriptions[mid]
        elif sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1, None