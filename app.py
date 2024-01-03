from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os
import io
import gradio as gr
import time
import dotenv
# !pip install git+https://github.com/huggingface/transformers.git

custom_prompt_template = """
You are a helpful,respectful, and honest assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant based on given user's query. 
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
Query: {query}
You just return the helpful answer.
Helpful Answer:
"""

# custom_prompt_template = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant."

# custom_prompt_template = """
# You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant based on given user's query. 
# You should treat people from different socioeconomic statuses, sexual orientations, religions, races, physical appearances, nationalities, gender identities, disabilities, and ages equally.
# When you do not have sufficient information, you should choose the unknown option, rather than making assumptions based on our stereotypes.
# Query: {query}
# You just return the helpful answer.
# Helpful Answer:
# """

# custom_prompt_template = """
# You are an AI Coding Assitant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
# Query: {query}

# You just return the helpful code.
# Helpful Answer:
# """

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
    input_variables=['query'])
    return prompt

# model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
# model_basename = "llama-2-13b-chat.ggmlv3.q8_0.bin" # the model is in bin format

# from huggingface_hub import hf_hub_download

# model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("vuongvd/elfbar-chat")
# model = AutoModelForCausalLM.from_pretrained("vuongvd/elfbar-chat")

# dotenv.load_dotenv('.env')
# HF_TOKEN = os.getenv('HF_TOKEN', default='')
# from huggingface_hub import login
# login(HF_TOKEN)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = "vuongvd/elfbar-chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

#Loading the model
def load_model():
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        # model = "meta-llama/Llama-2-7b-chat-hf",
        # model = "data/llama-2-7b-chat-ggml",
        # model_path=model_path,
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.2,
        repetition_penalty = 1.13
    )

    return llm

print(load_model())

def chain_pipeline():
    llm = load_model()
    qa_prompt = set_custom_prompt()
    qa_chain = LLMChain(
        prompt=qa_prompt,
        llm=llm
    )
    return qa_chain

llmchain = chain_pipeline()

def bot(query):
    llm_response = llmchain.run({"query": query})
    return llm_response

with gr.Blocks(title='Llama 2 Chatbot') as chat:
    gr.Markdown("# Llama 2 Chatbot")
    chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = bot(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

chat.launch()
