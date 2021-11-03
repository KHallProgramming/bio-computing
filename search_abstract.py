# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:35:10 2021

@author: k.hall@tees.ac.uk
"""

import numpy as np
import pandas as pd
import webbrowser
import functools
from IPython.core.display import display, HTML
from nltk import PorterStemmer
pd.options.mode.chained_assignment = None

### Phase 1: filter dataset for covid-related articles ###

# Filter dataset for articles containing covid related words appearing in abstract
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    dfe = df[df['abstract'].str.contains('pandemic')]
    frames=[dfa, dfb, dfc, dfd, dfe]
    df = pd.concat(frames)
    df = df.drop_duplicates(subset='title', keep="first")
    return df

# Import dataset
df = pd.read_csv('metadata.csv', 
                 usecols=['title','journal','abstract','authors','doi',
                          'publish_time','sha'])
print (df.shape)
# Drop duplicate entries
df = df.drop_duplicates()
# drop NANs 
df = df.fillna('no data provided')
df = df.drop_duplicates(subset='title', keep="first")
# Date filter
### df = df[df['publish_time'].str.contains('2020')]
# Convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
# Show 5 lines of the new dataframe
df = search_focus(df)
print(df.shape)
df.head()

### Phase 2: Filter by search terms ###

# Function to stem keyword list into a common base word
def stem_words(words):
    stemmer = PorterStemmer()
    singles = []
    for w in words:
        singles.append(stemmer.stem(w))
    return singles

def search_dataframe(df, search_words):
    search_words = stem_words(search_words)
    df1 = df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) 
                                               for s in search_words))]
    return df1

# Analyze search results for relevance with word count / abstract length
def search_relevance(rel_df, search_words):
    rel_df['score']=""
    search_words = stem_words(search_words)
    for index, row in rel_df.iterrows():
        abstract = row['abstract']
        result = abstract.split()
        len_abstract = len(result)
        score = 0
        for word in search_words:
            score = score + result.count(word)
        final_score = (score/len_abstract)
        rel_score = score * final_score
        rel_df.loc[index, 'score'] = rel_score
    rel_df = rel_df.sort_values(by = ['score'], ascending = False)
    return rel_df

# Function to get best sentences from the search results
def get_sentences(df1, search_words):
    df_table = pd.DataFrame(columns = ["pub_date", "authors", "title", 
                                       "excerpt", "rel_score"])
    search_words = stem_words(search_words)
    for index, row in df1.iterrows():
        pub_sentence = ''
        sentences_used = 0
        # Break apart the abstract to sentence level
        sentences = row['abstract'].split('. ')
        # Loop through the sentences of the abstract
        highligts= []
        for sentence in sentences:
            # Missing lets the system know if all the words are in the sentence
            missing = 0
            # Loop through the words of sentence
            for word in search_words:
                # If keyword missing change missing variable
                if word not in sentence:
                    missing = 1
            # After all sentences processed show the sentences not missing keywords
            if missing == 0 and len(sentence) < 1000 and sentence != '':
                sentence = sentence.capitalize()
                if sentence[len(sentence)-1] != '.':
                    sentence = sentence + '.'
                pub_sentence = pub_sentence + '<br><br>' + sentence
        if pub_sentence != '':
            sentence = pub_sentence
            sentences_used = sentences_used + 1
            authors = row["authors"].split(" ")
            link = row['doi']
            title = row["title"]
            score = row["score"]
            linka = 'https://doi.org/' + link
            linkb = title
            sentence = '<p fontsize=tiny" align="left">' + sentence + '</p>'
            final_link = '<p align="left"><a href="{}">{}</a></p>'.format(linka, linkb)
            to_append = [row['publish_time'], authors[0] + ' et al.',
                         final_link, sentence, score]
            df_length = len(df_table)
            df_table.loc[df_length] = to_append
    return df_table
    
### Main Search ###

terms = input("Enter search terms, seperate with space: ")
terms = terms.split()
search=[terms]
q = 0
for search_words in search:
    
    # Search the dataframe for all words
    df1 = search_dataframe(df, search_words)

    # Analyze search results for relevance 
    df1 = search_relevance(df1, search_words)

    # Get best sentences
    df_table = get_sentences(df1, search_words)
    
    length = df_table.shape[0]
    df_table = df_table.drop(['rel_score'], axis=1)
    # Convert dataframe to HTML
    df_table = HTML(df_table.to_html(escape = False, index = False))
    
    # Open table in browser
    if length < 1:
        print ("No reliable answer could be located in the literature")
    else:
        display(df_table)
        with open("df_table.html", "w", encoding='utf-8') as file:
            file.write(df_table.data)
        url = 'file:///C:/Users/Karl/Documents/archive/df_table.html'
        webbrowser.open(url, new = 2)
    q = q + 1
print ('Done')