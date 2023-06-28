from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, TFAutoModelForSeq2SeqLM,  T5Tokenizer
import tensorflow as tf
import streamlit as st
import PyPDF2 as pypdf
import time

@st.cache_data
def read_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def model_func(question,context):

    # koristimo pretrenirani model, jos uvek nije fine-tunovan
    model_name = "deepset/roberta-base-squad2"
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
    #model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ucitavanje tokenizatora
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenizacija ulaza 
    inputs = tokenizer(question, context, return_tensors="tf")

    # predikcije
    outputs = model(inputs)

    # odredjivanje pozicije pocetka i kraja odgovora
    start_logits = outputs.start_logits.numpy()[0]
    end_logits = outputs.end_logits.numpy()[0]
    start_index = int(tf.argmax(start_logits))
    end_index = int(tf.argmax(end_logits))

    # odgovor se vraca iz prosledjenog konteksta 
    answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index : end_index + 1])
    )
    st.write(answer)

def main():

    st.title('Sova demo app')
    uploaded_file = st.file_uploader("Choose a file", type=["txt","pdf"])
    user_input = st.text_input("Enter your question", "")
    
    if uploaded_file is not None and st.button("run model"):
        text = read_pdf(uploaded_file)
        model_func(user_input,text)
    else:
        st.text = "Please choose a file to load"

if __name__ == "__main__":
    main()




