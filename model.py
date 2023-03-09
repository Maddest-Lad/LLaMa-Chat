from gptq import *
from modelutils import *
from quant import *
from transformers import LLaMAConfig, LLaMAForCausalLM, LLaMATokenizer
from zeroShot.models.quant import *

modelname = 'llama-30b-hf'


def load_quant4(model, checkpoint):
    config = LLaMAConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LLaMAForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant4(model, layers)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


def load_quant(model, checkpoint, wbits):
    from transformers import LLaMAConfig, LLaMAForCausalLM
    config = LLaMAConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LLaMAForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load',
        default='llama-30b-hf'
    )
    parser.add_argument(
        'load', type=str,
        help='Load quantized model.'
    )

    prompt = "YOUR PROMPT GOES HERE"

    args = parser.parse_args()

    model = load_quant(args.model, args.load, 4).to('cuda')
    tokenizer = LLaMATokenizer.from_pretrained(args.model)
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.cuda() for k, v in batch.items()}
    generated = model.generate(batch["input_ids"], do_sample=True, use_cache=True, repetition_penalty=1.1,
                               max_length=2048, max_new_tokens=512, temperature=0.56, top_p=0.95, top_k=10)

    print(tokenizer.decode(generated[0]))
