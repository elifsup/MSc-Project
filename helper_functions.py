from dotenv import load_dotenv
load_dotenv('thesis.env')
import os
from openai import OpenAI
import openai
import anthropic
import voyageai
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import ast
import re
import pickle
import time

df_test = pd.read_csv('test_dataset.csv')
df_test['embedding'] = df_test['embedding'].apply(lambda x: np.array(ast.literal_eval(x))) # preprocess


def load_text_files(directory):
    """
    function to create a dictionary whose keys are the source document
    identifier and the values are the texts of the source documents
    """
    text_files = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                # Remove the '.txt' extension from the filename
                key_name = filename.rstrip('.txt')
                text_files[key_name] = file.read()
    return text_files


source_docs_dict = load_text_files('source_documents')

vo = voyageai.Client()

def get_embedding(text, model = 'voyage-law-2'):
    embed = vo.embed([text], model=model)
    return embed.embeddings[0] 


def choose_in_context_examples(clause_to_be_tested, df ,n_samples):
    """ 
    Uses the embeddings and chooses the closest n_samples (NOT PER CLASS)
    according to cosine similarity.

    clause_to_be_tested = test clause
    df = the dataframe in-context examples are taken from
    n_samples = no of in-context examples (not per class)
    
    """    
    ic_clauses = df['clause'].tolist()
    ic_risks = df['ground_truth_label'].tolist()
    ic_embeddings = np.vstack(df['embedding'].values)
    ic_representing = df['representing'].tolist()
    ic_contract_types = df['contract_type'].tolist()
    
    
    clause_embed_ = df_test.loc[df_test['clause'] == clause_to_be_tested, 'embedding']
    clause_embed = clause_embed_.values[0]  # Use preprocessed embedding directly
    
    
    similarities = np.dot(ic_embeddings, clause_embed)
    retrieved_ids = np.argsort(similarities)[-n_samples:][::-1]
    retrieved_ids = retrieved_ids.tolist()
    
    
    relevant_clauses = [ic_clauses[i] for i in retrieved_ids]
    relevant_risks = [ic_risks[i] for i in retrieved_ids]
    relevant_representing = [ic_representing[i] for i in retrieved_ids]
    relevant_contract_types = [ic_contract_types[i] for i in retrieved_ids]


    return relevant_clauses, relevant_risks, relevant_representing, relevant_contract_types




def choose_in_context_examples_2(clause_to_be_tested, df, n_samples):
    """ 
    Uses the embeddings and chooses the closest n_samples (PER CLASS)
    according to cosine similarity.

    clause_to_be_tested = test clause
    df = the dataframe in-context examples are taken from
    n_samples = no of in-context examples per class
    
    """
    # split the df according to risk
    red_flag_df = df[df['ground_truth_label'] == 'red flag']
    potential_issue_df = df[df['ground_truth_label'] == 'potential issue']

    # list the clauses (to retrieve later)
    clauses_red_flag = red_flag_df['clause'].tolist()
    clauses_potential_issue = potential_issue_df['clause'].tolist()

    # list the risks
    risks_red_flag = red_flag_df['ground_truth_label'].tolist()
    risks_potential_issue = potential_issue_df['ground_truth_label'].tolist()

    # list the representing
    representing_red_flag = red_flag_df['representing'].tolist()
    representing_potential_issue = potential_issue_df['representing'].tolist()

    # list the contract types
    contract_type_red_flag = red_flag_df['contract_type'].tolist()
    contract_type_potential_issue = potential_issue_df['contract_type'].tolist()

    # list the embeddings 
    test_embeddings_red_flag = np.vstack(red_flag_df['embedding'].values)
    test_embeddings_potential_issue = np.vstack(potential_issue_df['embedding'].values)
    
    # get the embedding of the clause to be tested
    clause_embed_ = df_test.loc[df_test['clause'] == clause_to_be_tested, 'embedding']
    clause_embed = clause_embed_.values[0]  # Use preprocessed embedding directly
 
    # find similarity scores
    similarities_red_flag = np.dot(test_embeddings_red_flag, clause_embed)
    similarities_potential_issue = np.dot(test_embeddings_potential_issue, clause_embed)

    # indexes for the top n_samples for each risk group
    retrieved_ids_red_flag = np.argsort(similarities_red_flag)[-n_samples:][::-1]
    retrieved_ids_potential_issue = np.argsort(similarities_potential_issue)[-n_samples:][::-1]

    # converting these to lists
    retrieved_ids_red_flag = retrieved_ids_red_flag.tolist()
    retrieved_ids_potential_issue = retrieved_ids_potential_issue.tolist()

    # the corresponding clauses
    relevant_clauses_red_flag = [clauses_red_flag[i] for i in retrieved_ids_red_flag]
    relevant_clauses_potential_issue = [clauses_potential_issue[i] for i in retrieved_ids_potential_issue]
    relevant_clauses = relevant_clauses_red_flag + relevant_clauses_potential_issue
    # trying to see if order effects anything
    #alt_relevant_clauses =  relevant_clauses_potential_issue +  relevant_clauses_red_flag

    # corresponding risks
    relevant_risks_red_flag = [risks_red_flag[i] for i in retrieved_ids_red_flag]
    relevant_risks_potential_issue = [risks_potential_issue[i] for i in retrieved_ids_potential_issue]
    relevant_risks = relevant_risks_red_flag + relevant_risks_potential_issue

    # corresponding representing
    relevant_representing_red_flag = [representing_red_flag[i] for i in retrieved_ids_red_flag]
    relevant_representing_potential_issue = [representing_potential_issue[i] for i in retrieved_ids_potential_issue]
    relevant_representing = relevant_representing_red_flag + relevant_representing_potential_issue
  
    # corresponding contract types
    relevant_contract_type_red_flag = [contract_type_red_flag[i] for i in retrieved_ids_red_flag]
    relevant_contract_type_potential_issue = [contract_type_potential_issue[i] for i in retrieved_ids_potential_issue]
    relevant_contract_type = relevant_contract_type_red_flag + relevant_contract_type_potential_issue  
   
    return relevant_clauses, relevant_risks, relevant_representing, relevant_contract_type
    


def prepare_fewshot_system_content(init_system_content, ic_clauses, ic_risks, ic_representing, ic_contract_type):
    init_system_content += '\nHere are some examples to help you: \n '
    for clause, risk, representing, contract_type in zip(ic_clauses, ic_risks, ic_representing, ic_contract_type):
        init_system_content += f"Information: The type of this contract is {contract_type}. Representing side is the {representing}. The governing law is England and Wales. \n Clause: {clause} \n Answer: {risk} \n "
    return init_system_content



def prepare_fewshot_prompt(ic_clauses, ic_risks, ic_representing, ic_contract_type):
    prompt = "" # init with empty string
    for clause, risk, representing, contract_type in zip(ic_clauses, ic_risks, ic_representing, ic_contract_type):
        prompt += f"Information: The type of this contract is {contract_type}. Representing side is the {representing}. The governing law is England and Wales. \nClause: {clause} \nAnswer: {risk} \n"
    return prompt


def append_context(text, clause, contract_type, representing): # the text in here is either the system message or the prompt
    text += f' The type of this contract is {contract_type}. Representing side is the {representing}. The governing law is England and Wales.'
    return text


# for when we append the whole contract in the system message
def append_context_and_source(text, contract_type, representing, source): # the text in here is either the system message or the prompt
    text += f' The type of this contract is {contract_type}. Representing side is the {representing}. The governing law is England and Wales.\
        \nThe contract this clause is taken from is: \n {source_docs_dict[source]}'
    return text


# for when we append the whole contract in the prompt
def appending_contract_to_prompt(clause,source):
    return f"The clause is: \n{clause}. \nThe contract this clause is taken from is: \n{source_docs_dict[source]}"



def sample_in_context_examples(df, n_samples, random_state=42):
    """
    randomly choosing n in-context examples for each risk type from 
    the training dataset (balanced_dataset) and returning the lists
    """
    risk_column = 'ground_truth_label'
    clause_column = 'clause'

    sampled_df = df.groupby(risk_column)[[clause_column, risk_column,'representing','contract_type']].apply(lambda x: x.sample(n_samples, random_state=random_state)).reset_index(drop=True)
    sampled_clauses = sampled_df[clause_column].tolist()
    sampled_risks = sampled_df[risk_column].tolist()
    sampled_representing = sampled_df['representing'].tolist()
    sampled_contract_type = sampled_df['contract_type'].tolist()

    return sampled_clauses, sampled_risks, sampled_representing, sampled_contract_type


