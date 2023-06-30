from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, TFAutoModelForSeq2SeqLM,  T5Tokenizer,TFT5ForConditionalGeneration
import tensorflow as tf
import streamlit as st
import PyPDF2 as pypdf

@st.cache_data
def read_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

def t5_model_func(question,context):

    # koristimo pretrenirani model, jos uvek nije fine-tunovan
    model_name = "google/flan-t5-base"
    model = TFT5ForConditionalGeneration.from_pretrained(model_name)
    #model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ucitavanje tokenizatora
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenizacija ulaza 
    input_text = question + " " + context
    #inputs = tokenizer(input_text,return_tensors="tf")

    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write(answer)

def roberta_model_func(question,context):

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




