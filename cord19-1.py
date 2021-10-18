# -*- coding: utf-8 -*-
"""
Created on Thur Oct 14 14:53:33 2021

@author: k.hall@tees.ac.uk

Visualise and Export the most common genes, drugs and gene regulation 
and associated search functions.
"""

import pandas as pd
import numpy as np
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
import ast
import os

# Dataset import and setup into pandas
df = pd.read_csv('C:/Users/Karl/Documents/archive/input/final.csv', error_bad_lines=False, encoding='ISO-8859-1',
                 index_col=0)

df['Biomedical_Entities'] = df['Biomedical_Entities'].apply(ast.literal_eval)
df['genes'] = df['Genes'].apply(ast.literal_eval)
df['drugs'] = df['Drugs'].apply(ast.literal_eval)

# Show dataset structure; genes and drugs arrays
df_head = df.head()
print(df.head(2))

# Make copy of dataframe for search functionality
df_search = df

# Remove '#' in selection and df = ... to apply the search

# You can create a search of papers that contain keywords, add them in the list
selection = ['liver']
df_search = df_search[pd.DataFrame(df_search.Biomedical_Entities.tolist()).isin(selection).any(1)]

# You can create a search of genes of interest, add them in the list
# selection = ['vim']
# df_search = df_search[pd.DataFrame(df_search.genes.tolist()).isin(selection).any(1)]

# You can create a search of drugs of interest, add them in the list

# selection = ['atp']
# df_search = df_search[pd.DataFrame(df_search.drugs.tolist()).isin(selection).any(1)]

# You can create a search to only look at preprints or peer reviewed papers
# selection = ['Peer-Review']
# df_search = df[pd.DataFrame(df.preprint.tolist()).isin(selection).any(1)]

df_search


# Convert most frequent words to dataframe eg. for plots
def get_top_n_genes(corpus, n=20):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


# Check for empty list
def check_if_empty(list_of_lists):
    lst = []

    for elem in list_of_lists:
        if len(elem) == 0:
            lst.append(False)
        else:
            lst.append(True)
    if True in lst:
        return True
    else:
        return False


# Visualise a column of dataframe using altair of top column lines
def visualise_data(df, col, n=20):
    if n == 0:
        n = 20
        print("You asked for 0 genes, so here are the top 20.")

    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
        flat_list = [item for sublist in lst for item in sublist]
        top_words = get_top_n_genes(lst, n)
        top_df = pd.DataFrame(top_words)
        top_df.columns = [col, "Freq_gene"]
        plot = alt.Chart(top_df).mark_bar().encode(
            x=alt.X(col, sort='-y', axis=alt.Axis(title=col)),
            y=alt.Y('Freq_gene', axis=alt.Axis(
                title='Frequency Count'))).properties(
            title="Frequency of the top {} {}".format(
                str(n), col), width=200)

        return plot
    else:
        return False


# Visualise top genes and drugs
def visualise_multiple_columns(df, lst, n = 20):
    plots = []
    for el in lst:
        plt = visualise_data(df, el, n)
        if plt != False:
            plots.append(plt)
            # Opens in browser with altair viewer library
            plt.show()

    a = plots[0]
    for i in range(1, len(plots)):
        a = a | plots[i]
    return a


visualise_multiple_columns(df, ['Genes', 'Drugs'], 20)


# Return dataframe with frequency of top drugs
def get_df(df, col, n=20):
    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
        flat_list = [item for sublist in lst for item in sublist]
        top_words = get_top_n_genes(lst, n)
        top_df = pd.DataFrame(top_words)
        top_df.columns = [col, "Frequency"]

        return top_df
    else:
        return False


# Use these for Genes or Drugs, change n to increase number of top drugs/genes
top_genes_df = get_df(df, 'Genes', n=20)
top_drugs_df = get_df(df, 'Drugs', n=20)


# To export these as a csv file
# top_genes_df.to_csv('top_genes_df.csv', index = False)
# top_drugs_df.to_csv('top_drugs_df.csv', index = False)

# Visualises a column of the df using altair of the top items in a column
def visualise_data_up_reg(df, col, n=20):
    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
        top_words = get_top_n_genes(lst, n)
        top_df = pd.DataFrame(top_words)
        top_df.columns = [col, "Freq_gene"]
        plot = alt.Chart(top_df).mark_bar().encode(
            x=alt.X(col, sort='-y', axis=alt.Axis(title=col)),
            y=alt.Y('Freq_gene', axis=alt.Axis(title='Frequency Count'))
        ).properties(title="Frequency of the top {} {}".format(
            str(n), col), width=200)

        return plot
    else:
        return False


# Visualise regulation
# Regulation is calculated using nlp techniques. 
# Currently it is a simplistic based on dentification of common terms used to 
# describe gene regulation in the surrounding text of an identified gene.
def visualise_multiple_columns_reg(df, lst, n=20):
    if n == 0:
        n = 20
        print('Default: Showing top 20.. ')

    plots = []
    for el in lst:
        plt = visualise_data_up_reg(df, el, n)
        if plt != False:
            plots.append(plt)
            plt.show()

    a = plots[0]
    for i in range(1, len(plots)):
        a = a | plots[i]

    return a


visualise_multiple_columns_reg(df, ['Genes_Upregulated', 'Genes_Downregulated', 'Genes_Nonregulated'], 20)


# This function returns a df with the frequency of the regulation
def get_df(df, col, n=20):
    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
        top_words = get_top_n_genes(lst, n)
        top_df = pd.DataFrame(top_words)
        top_df.columns = [col, "Freq_gene"]

        return top_df
    else:
        return False


# Use it for regulation
top_words_df = get_df(df, 'Genes_Upregulated', n=20)
top_words_df.head()
