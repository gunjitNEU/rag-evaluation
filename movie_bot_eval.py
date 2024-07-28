import streamlit as st
import ollama
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from rouge_score import rouge_scorer
import re
import time

# Initialize LLM and other components
llm = Ollama(model='llama3-chatqa')
embeddings = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Act like a movie recommender with the information you have. 
whenever you recommend a movie give a little description about it as well.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

qa_data = [
    ("What is 'Taxi Blues' about?", "Taxi Blues is a Russian film directed by Pavel Lungin."),
    ("What is 'The Hunger Games' about?", "The Hunger Games is a dystopian science fiction film based on the novel by Suzanne Collins."),
    ("What is 'Narasimham' about?", "Narasimham is a Malayalam film directed by Shaji Kailas, starring Mammootty."),
    ("What is 'The Lemon Drop Kid' about?", "The Lemon Drop Kid is a 1951 comedy film starring Bob Hope."),
    ("What is 'A Cry in the Dark' about?", "A Cry in the Dark is a 1988 Australian film starring Meryl Streep and Sam Neill, based on the true story of Lindy Chamberlain."),
    ("What is 'End Game' about?", "End Game is a 2006 South African film directed by Pete Travis."),
    ("What is 'Dark Water' about?", "Dark Water is a 2005 American horror film directed by Walter Salles and based on the Japanese film of the same name."),
    ("What is 'Sing' about?", "Sing is a 2016 animated musical film produced by Illumination Entertainment."),
    ("What is 'Meet John Doe' about?", "Meet John Doe is a 1941 American film directed by Frank Capra, starring Gary Cooper and Barbara Stanwyck."),
    ("What is 'Ghost In The Noonday Sun' about?", "Ghost In The Noonday Sun is a 1973 film starring Peter Sellers."),
]


# Updated sample questions and answers
sample_questions = [i[0] for i in qa_data]


sample_answers = [i[1] for i in qa_data]


# Initialize metrics storage
y_true = []
y_pred = []
context_retrieved = []
context_relevant = []
context_entities = []
latencies = []

# Title of the app
st.title("Movie Master Chatbot")

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input for user query
query = st.text_input("Ask for your movie recommendation:")

# Define helper functions for context evaluation
def evaluate_response(actual_answer, generated_response):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(actual_answer, generated_response)
    return scores

def context_metrics(query, retrieved_contexts, relevant_contexts):
    true_positives = sum([1 for ctx in retrieved_contexts if ctx in relevant_contexts])
    precision = true_positives / len(retrieved_contexts) if retrieved_contexts else 0
    recall = true_positives / len(relevant_contexts) if relevant_contexts else 0
    return precision, recall

def context_relevance_metrics(retrieved_contexts, relevant_contexts):
    relevance_scores = []
    for context in retrieved_contexts:
        score = sum([1 for rel_ctx in relevant_contexts if rel_ctx in context])
        relevance_scores.append(score)
    return np.mean(relevance_scores) if relevance_scores else 0

def context_entity_recall(retrieved_entities, relevant_entities):
    true_positives = sum([1 for ent in retrieved_entities if ent in relevant_entities])
    recall = true_positives / len(relevant_entities) if relevant_entities else 0
    return recall

def noise_robustness(query, generated_response):
    return 1 if re.search(r'\b(noise|irrelevant|nonsense)\b', query, re.IGNORECASE) else 0

def generation_metrics(query, generated_response, actual_answer):
    # Faithfulness
    faithfulness = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True).score(actual_answer, generated_response)
    
    # Answer Relevance (assuming exact match for simplicity)
    relevance = 1 if generated_response.strip().lower() == actual_answer.strip().lower() else 0
    
    # Information Integration (using ROUGE scores as proxy)
    integration = np.mean([faithfulness['rouge1'].fmeasure, faithfulness['rougeL'].fmeasure])
    
    # Counterfactual Robustness (test against a counterfactual query)
    counterfactual = 1 if re.search(r'\b(counterfactual|contradictory)\b', query, re.IGNORECASE) else 0
    
    # Negative Rejection (simple negative test)
    negative_rejection = 1 if re.search(r'\b(irrelevant|negative|inappropriate)\b', generated_response, re.IGNORECASE) else 0
    
    return faithfulness, relevance, integration, counterfactual, negative_rejection

# Evaluate sample questions
for i, question in enumerate(sample_questions):
    start_time = time.time()  # Start timer
    response = qa_chain({"query": question})
    end_time = time.time()    # End timer
    
    generated_response = response['result']
    latency = end_time - start_time  # Calculate latency
    
    # Mock context and entity data (replace with real context and entities)
    retrieved_contexts = ["Inception is a science fiction film with action and thriller elements.",
                          "The Shawshank Redemption was directed by Frank Darabont.",
                          "Pulp Fiction was released in 1994.",
                          "The Dark Knight stars Christian Bale as Batman.",
                          "The Matrix is about Neo fighting against a simulated reality.",
                          "Parasite won Best Picture, Director, Original Screenplay, and International Feature.",
                          "Harry Potter has eight movies.",
                          "Forrest Gump is played by Tom Hanks.",
                          "The Lost World: Jurassic Park is the sequel.",
                          "Gladiator is set in ancient Rome."]
    relevant_contexts = [sample_answers[i]]
    retrieved_entities = re.findall(r'\b(\w+)\b', generated_response)
    relevant_entities = re.findall(r'\b(\w+)\b', sample_answers[i])
    
    # Append to metrics storage
    y_true.append(sample_answers[i])
    y_pred.append(generated_response)
    context_retrieved.append(retrieved_contexts)
    context_relevant.append(relevant_contexts)
    context_entities.append(retrieved_entities)
    latencies.append(latency)
    
    # Evaluate the response
    evaluation_scores = evaluate_response(sample_answers[i], generated_response)
    precision, recall = context_metrics(question, retrieved_contexts, relevant_contexts)
    relevance = context_relevance_metrics(retrieved_contexts, relevant_contexts)
    entity_recall = context_entity_recall(retrieved_entities, relevant_entities)
    noise_robust = noise_robustness(question, generated_response)
    faithfulness, answer_relevance, integration, counterfactual, negative_rejection = generation_metrics(question, generated_response, sample_answers[i])
    
    # Update conversation history
    st.session_state.history.append({'role': 'user', 'content': question})
    st.session_state.history.append({'role': 'bot', 'content': generated_response})
    
    # Display evaluation scores
    st.write(f"Question: {question}")
    st.write(f"Actual Answer: {sample_answers[i]}")
    st.write(f"Generated Response: {generated_response}")
    st.write(f"ROUGE Scores: {evaluation_scores}")
    st.write(f"Context Precision: {faithfulness['rouge1'].precision}")
    st.write(f"Context Recall: {faithfulness['rouge1'].recall}")
    st.write(f"Context Relevance: {relevance}")
    st.write(f"Context Entity Recall: {entity_recall}")
    st.write(f"Noise Robustness: {noise_robust}")
    st.write(f"Faithfulness: {faithfulness['rouge1'].fmeasure}")
    st.write(f"Answer Relevance: {answer_relevance}")
    st.write(f"Information Integration: {integration}")
    st.write(f"Counterfactual Robustness: {counterfactual}")
    st.write(f"Negative Rejection: {negative_rejection}")
    st.write(f"Latency: {latency:.2f} seconds")
    st.write("-" * 50)

# Calculate precision, recall, and F1 score
def calculate_metrics(y_true, y_pred):
    y_true_binary = [1 if ans in sample_answers else 0 for ans in y_true]
    y_pred_binary = [1 if ans in sample_answers else 0 for ans in y_pred]
    
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return precision, recall, f1

# Calculate metrics
precision, recall, f1 = calculate_metrics(y_true, y_pred)

# Display metrics
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Display the conversation history
for message in st.session_state.history:
    if message['role'] == 'user':
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")
