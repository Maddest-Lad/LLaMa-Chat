from flask import Flask, request
import json
from model import *
from transformers import LLaMATokenizer

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        prompt = json.loads(request.data)['prompt']
        model = load_quant(args.model, args.load, 4).to('cuda')
        tokenizer = LLaMATokenizer.from_pretrained(args.model)
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        batch = {k: v.cuda() for k, v in batch.items()}
        generated = model.generate(batch["input_ids"], do_sample=True, use_cache=True, repetition_penalty=1.1,
                                   max_length=2048, max_new_tokens=512, temperature=0.56, top_p=0.95, top_k=10)
        response = {'generated_text': tokenizer.decode(generated[0])}
    except Exception as e:
        response = {'error': str(e)}
    return json.dumps(response)


if __name__ == '__main__':
    app.run()
