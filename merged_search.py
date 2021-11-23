# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:14:46 2021

@author: Karl
"""

import pandas as pd
from nltk import PorterStemmer
import functools
from IPython.core.display import display, HTML
import webbrowser
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
import ast

### Functions - Part 1 ###

# Filter dataset for articles containing covid related words appearing in abstract
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    dfe = df[df['abstract'].str.contains('pandemic')]
    frames = [dfa, dfb, dfc, dfd, dfe]
    df = pd.concat(frames)
    dfa = df.drop_duplicates(subset = 'title', keep = 'first')
    return dfa

# Function that uses stemmer to stem the words (root meanings)
def stem_words(words):
    stemmer = PorterStemmer()
    singles = []
    for w in words:
        singles.append(stemmer.stem(w))
    return singles

# Function to search dataset for the search terms
def search_dataframe(df, search_words):
    search_words = stem_words(search_words)
    df1 = df[functools.reduce(lambda a, b: a & b, (df['abstract'].str.contains(s) 
                                               for s in search_words))]
    return df1

# Function to analyze search results for relevance using a score value
# with word count / abstract length

def search_relevance(rel_df, search_words):
    rel_df['score'] = ""
    search_words = stem_words(search_words)
    for index, row in rel_df.iterrows():
        abstract = row['abstract']
        result = abstract.split()
        len_abstract = len(result)
        score = 0
        for word in search_words:
            score = score + result.count(word)
        final_score = (score / len_abstract)
        rel_score = score * final_score
        rel_df.loc[index, 'score'] = rel_score
    rel_df = rel_df.sort_values(by = ['score'], ascending = False)
    return rel_df

# Table Builder #
def get_sentences(df1, search_words):
    df_table = pd.DataFrame(columns = ['pub_date', 'authors', 'title', 
                                       'excerpt', 'rel_score', 'drugs', 'genes', 
                                       'genes_upregulated', 'genes_downregulated', 
                                       'genes_nonregulated'])
    search_words = stem_words(search_words)
    for index, row in df1.iterrows():
        pub_sentence = ''
        sentences_used = 0
        # Break apart the abstract to sentence level
        sentences = row['abstract'].split('. ')
        # Loop through the sentences of the abstract
        for sentence in sentences:
            # Missing lets the system know if all the words are in the sentence
            missing = 0
            # Loop through the words of sentence
            for word in search_words:
                # If keyword missing change missing variable flag
                if word not in sentence:
                    missing = 1
            # After all sentences processed show the sentences not missing keywords
            if missing == 0 and len(sentence) < 1000 and sentence != '':
                sentence = sentence.capitalize()
                if sentence[len(sentence) - 1] != '.':
                    sentence = sentence + '.'
                pub_sentence = pub_sentence + '<br><br>' + sentence
        # Build the table
        if pub_sentence != '':
            sentence = pub_sentence
            sentences_used = sentences_used + 1
            authors = row['authors'].split(' ')
            link = row['doi']
            title = row['title']
            score = row['score']
            drugs = row['Drugs']
            genes = row['Genes']
            genes_upregulated = row['Genes_Upregulated']
            genes_downregulated = row['Genes_Downregulated']
            genes_nonregulated = row['Genes_Nonregulated']
            link_a = 'https://doi.org/' + link
            link_b = title
            sentence = '<p fontsize = tiny" align="left">' + sentence + '</p>'
            final_link = '<p align = "left"><a href="{}">{}</a></p>'.format(link_a, link_b)
            to_append = [row['publish_time'], authors[0] + ' et al.',
                         final_link, sentence, score, drugs, genes, genes_upregulated, genes_downregulated, genes_nonregulated]
            df_length = len(df_table)
            df_table.loc[df_length] = to_append
    return df_table

### Main program Part 1 - Data search tool ###

# Import metadata file using only important columns
df_meta = pd.read_csv('metadata.csv', 
                 usecols = ['cord_uid','title','journal','abstract','authors','doi',
                         'sha'])
# Import genes file using only important columns
df_genes = pd.read_csv('final.csv')
# Merge the two datasets together using cord_uid as a pivot
merged_df = pd.merge(df_meta, df_genes, on = 'cord_uid')
# Check for null values
print(merged_df.isna().sum())
# Drop rows in the merged data that contain null values we care about
merged_df = merged_df.dropna(subset = ['cord_uid', 'title', 'abstract', 
                                       'authors', 'Genes_Upregulated', 
                                       'Genes_Downregulated', 'Genes_Nonregulated'])
# Double check null values
print(merged_df.isna().sum())
# Fill other fields with No data provided: for less important fields
merged_df = merged_df.fillna('No data provided')
# Remove duplicate entries with tyhe same title and/or abstract 
merged_df = merged_df.drop_duplicates(subset = ['title', 'abstract'], keep = "first")
# Normalise to lowercase
merged_df['abstract'] = merged_df['abstract'].str.lower() + merged_df['title'].str.lower()
# Search
merged_df = search_focus(merged_df)
print(merged_df.shape)

# Enter search terms
terms = input('Enter search terms, seperate with spaces: ')
terms = terms.split()
search = [terms]
q = 0
for search_words in search:
    
    # Search the dataframe for all words
    df1 = search_dataframe(merged_df, search_words)
    # Analyze search results for relevance 
    df1 = search_relevance(df1, search_words)
    # Get best sentences
    df_table = get_sentences(df1, search_words)
    length = df_table.shape[0]
    df_table = df_table.drop(['rel_score'], axis = 1)
    # Convert dataframe to HTML
    df_table = HTML(df_table.to_html(escape = False, index = False))
    
    # Open table in browser
    if length < 1:
        print ('No reliable answer could be located in the repository')
    else:
        display(df_table)
        with open('df_table.html', 'w', encoding = 'utf-8') as file:
            file.write(df_table.data)
        url = 'file:///C:/Users/Karl/Documents/archive/df_table.html'
        webbrowser.open(url, new = 2)
    q = q + 1

### Main program Part 2 - Genes/Drugs Analysis ###

merged_df['genes'] = df1['Genes'].apply(ast.literal_eval)
merged_df['drugs'] = df1['Drugs'].apply(ast.literal_eval)

# Convert most frequent words to dataframe eg. for plots
def get_top_n_genes(corpus, n = 20):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0)
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
def visualise_data(df, col, n = 20):
    if n == 0:
        n = 20
        print("You asked for 0 genes, so here are the top 20.")

    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
        top_words = get_top_n_genes(lst, n)
        top_df = pd.DataFrame(top_words)
        top_df.columns = [col, "Freq_gene"]
        plot = alt.Chart(top_df).mark_bar().encode(
            x=alt.X(col, sort='-y', axis=alt.Axis(title=col)),
            y=alt.Y('Freq_gene', axis=alt.Axis(
                title='Frequency Count'))).properties(
            title="[Search terms: " + terms[0] + "] Frequency of the top {} {}".format(
                str(n), col), width = 200)

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


visualise_multiple_columns(df1, ['Genes', 'Drugs'], 20)


# Return dataframe with frequency of top drugs
def get_df(df, col, n = 20):
    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
        top_words = get_top_n_genes(lst, n)
        top_df = pd.DataFrame(top_words)
        top_df.columns = [col, "Frequency"]

        return top_df
    else:
        return False


# Use these for Genes or Drugs, change n to increase number of top drugs/genes
top_genes_df = get_df(df1, 'Genes', n = 20)
top_drugs_df = get_df(df1, 'Drugs', n = 20)


# To export these as a csv file
# top_genes_df.to_csv('top_genes_df.csv', index = False)
# top_drugs_df.to_csv('top_drugs_df.csv', index = False)

# Visualises a column of the df using altair of the top items in a column
def visualise_data_up_reg(df, col, n = 20):
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
        ).properties(title="[Search terms: "+ terms[0] +"] \n Frequency of the top {} {}".format(
            str(n), col), width=200)

        return plot
    else:
        return False


# Visualise regulation
# Regulation is calculated using nlp techniques. 
# Currently it is a simplistic based on identification of common terms used to 
# describe gene regulation in the surrounding text of an identified gene.
def visualise_multiple_columns_reg(df, lst, n = 20):
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


visualise_multiple_columns_reg(df1, ['Genes_Upregulated', 'Genes_Downregulated', 'Genes_Nonregulated'], 20)


# This function returns a df with the frequency of the regulation
def get_df(df, col, n = 20):
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


# Use this to print as a table, can change column to downregulated/nonregulated
top_words_df = get_df(df1, 'Genes_Upregulated', n = 20)
print(top_words_df.head())