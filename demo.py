from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, TFAutoModelForSeq2SeqLM,  T5Tokenizer,TFT5ForConditionalGeneration, DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf
import streamlit as st
import PyPDF2 as pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def read_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def calculate_similarity(question, answer):
    vectorizer = TfidfVectorizer()
    pair = [question, answer]
    tfidf_matrix = vectorizer.fit_transform(pair)

    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    similarity = similarity_matrix[0][0]

    return similarity

def roberta_model_generate_answer(question, segment, model, tokenizer):
    # Tokenizacija ulaza
    inputs = tokenizer(question, segment, return_tensors="tf")

    # Predikcije
    outputs = model(inputs)

    # Određivanje pozicije početka i kraja odgovora
    start_logits = outputs.start_logits.numpy()[0]
    end_logits = outputs.end_logits.numpy()[0]
    start_index = int(tf.argmax(start_logits))
    end_index = int(tf.argmax(end_logits))

    # Odgovor se dobija iz prosleđenog segmenta
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index: end_index + 1])
    )

    return answer

def t5_model_generate_answer(question, segment, model, tokenizer):
    # Tokenizacija ulaza
    input_text = question + " " + segment
    input_ids = tokenizer.encode(input_text, return_tensors='tf')

    # Generisanje odgovora
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

def roberta_model_func(question, context):
    # Koristimo pretrenirani model, još uvek nije fine-tjunovan
    model_name = "deepset/roberta-base-squad2"
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

    # Učitavanje tokenizatora
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Podela konteksta na delove sa stride-om od 256 tokena
    max_length = 512  # Maksimalna dužina konteksta
    stride = 256  # Stride za podelu konteksta
    context_parts = [context[i:i + max_length] for i in range(0, len(context), stride)]

    # Inicijalizacija promenljive za najbolji odgovor
    best_answer = None

    # Inicijalizacija promenljive za najbolju sličnost
    best_similarity = 0
    best_segment = context_parts[0]

    # Iteriranje kroz svaki deo konteksta
    for segment in context_parts:
        # Računanje sličnosti između pitanja i segmenta
        similarity = calculate_similarity(question, segment)

        # Ažuriranje najboljeg odgovora i sličnosti
        if similarity > best_similarity:
            best_similarity = similarity
            best_segment = segment
    
    best_answer = roberta_model_generate_answer(question, best_segment, model, tokenizer)

    st.write(best_answer)

def t5_model_func(question, context):
    # Koristimo pretrenirani model, još uvek nije fine-tjunovan
    model_name = "google/flan-t5-small"
    model = TFT5ForConditionalGeneration.from_pretrained(model_name)

    # Učitavanje tokenizatora
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Podela konteksta na delove sa stride-om od 256 tokena
    max_length = 512  # Maksimalna dužina konteksta
    stride = 256  # Stride za podelu konteksta
    context_parts = [context[i:i + max_length] for i in range(0, len(context), stride)]

    # Inicijalizacija promenljive za najbolju sličnost
    best_similarity = 0
    best_segment = context_parts[0]

    for segment in context_parts:
        # Računanje sličnosti između pitanja i segmenta
        similarity = calculate_similarity(question, segment)

        # Ažuriranje najboljeg odgovora i sličnosti
        if similarity > best_similarity:
            best_similarity = similarity
            best_segment = segment

    best_answer = t5_model_generate_answer(question, best_segment, model, tokenizer)
    st.write(best_answer)

models = {
    'T5': t5_model_func,
    'Roberta': roberta_model_func
}

def main():

    st.title('Sova demo app')

    selected_model = st.selectbox(
    'Which model would you like to use?',
    ('T5', 'Roberta'))
    st.write('You selected:', selected_model)
    model_func = models[selected_model]

    uploaded_file = st.file_uploader("Choose a file", type=["txt","pdf"])
    user_input = st.text_input("Enter your question", "")
    
    if uploaded_file is not None and st.button("run model"):
        text = read_pdf(uploaded_file)
        model_func(user_input, text)
    else:
        st.text = "Please choose a file to load"

if __name__ == "__main__":
    main()




