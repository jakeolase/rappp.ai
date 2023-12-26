from flask import Flask, render_template, request
from model import recommend_prog_lang_similar_to
from model import merge_sort
from model import binary_search
from model import cleaned_text
from model import get_keywords
from model import update_dataframe_and_calculate_similarity
from model import paper_update_dataframe_and_calculate_similarity
from model import paper_recommender
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', forms=False)

@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']

    if len(user_input) > 49:
        model_df = pd.read_csv("df_keyBert.csv")
        paper_df = pd.read_csv("paper_keyBert.csv")
        cleaned_result, keywords, prog_cosine_sim = update_dataframe_and_calculate_similarity(user_input, model_df)
        paper_cleaned_result, paper_keywords, paper_prog_cosine_sim = paper_update_dataframe_and_calculate_similarity(user_input, paper_df)

        recommendations, project_set = recommend_prog_lang_similar_to(cleaned_result, cosine_sim_mat=prog_cosine_sim, programs_list=model_df['Language'], projects=model_df['URL'])
         # Perform binary search and print results for each recommendation
        df_unique = pd.read_csv("prog_lang_unique.csv")
        all_results = []

        papers= paper_recommender(paper_cleaned_result, cosine_sim_mat=paper_prog_cosine_sim, papers_list=paper_df['title'], link_list=paper_df['url'], authors_list=paper_df['author'])
        # sort the list and descriptions using merge sort
        sorted_languages, sorted_descriptions = merge_sort(df_unique['Language'].tolist(), df_unique['Description'].tolist())

        for recommendation in recommendations:
            target_language = recommendation[0]

            # create a list to store the results for the current recommendation
            results_list = []

            # perform binary search on the sorted list
            result_index, result_description = binary_search(sorted_languages, sorted_descriptions, target_language)

            if result_index != -1:
                results_list.append(result_description)
            else:
                results_list.append(f"{target_language} not found in the list.")

            # append the results for the current recommendation to the overall results list
            all_results.append(results_list)


        project_names = []

        for project in project_set:
            if isinstance(project, str):
                match = re.search(r'https://github\.com/([^/]+)/([^/]+)', project)
                if match:
                    project_names.append(match.group(2))
            elif isinstance(project, list) and len(project) > 0 and isinstance(project[0], str):
                match = re.search(r'https://github\.com/([^/]+)/([^/]+)', project[0])
                if match:
                    project_names.append(match.group(2))

        combined = zip(recommendations, all_results)
        projects = list(zip(project_set, project_names))

        print(projects)
        # Display the result in the index.html
        return render_template('index.html', recommendations=recommendations, projects = projects, all_results=all_results, combined=combined, paper=papers, forms=True, user_input = user_input)
    else:
        return render_template('index.html', error="Minimum 50 characters required.", user_input = user_input)

if __name__ == '__main__':
    app.run(host='localhost', port = 3000, debug=True)
