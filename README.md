# Gradio Chatbot using Llama-2

## Create a virtual environment (optional but recommended) and activate it:

```bash
python -m venv venv
source venv/bin/activate  
```

## Install the required Python packages:

```bash
pip install -r requirements.txt
```

### If you use your private model in Huggingface

1. Change .env.example to .env and update HF_TOKEN 

2. Uncomment this validation code in *app.python*:
```
dotenv.load_dotenv('.env')
HF_TOKEN = os.getenv('HF_TOKEN', default='')
from huggingface_hub import login
login(HF_TOKEN)
```

## Usage

```bash
python run app.py
```

Url append in terminal (ex: *Running on local URL: http://127.0.0.1:7860*)
