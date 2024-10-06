from config import GPTConfig
from gpt import GPT
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_hf_weights(model: GPT):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    state_dict = model.state_dict()
    state_dict_hf = model_hf.state_dict()

    # Ensure model architectures match
    assert set(state_dict.keys()) == set(state_dict_hf.keys()), "The model architectures do not match!"

    # Transpose certain weights since we use `nn.Linear`
    transpose = ["attn.qkv_proj.weight", "attn.out_proj.weight", "ffn.fc1.weight", "ffn.fc2.weight"]
    for key in state_dict.keys():
        if any(layer in key for layer in transpose):
            state_dict[key] = state_dict_hf[key].T
        else:
            state_dict[key] = state_dict_hf[key]
    
    model.load_state_dict(state_dict)

def main():
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    cfg = GPTConfig()
    model = GPT(cfg)
    load_hf_weights(model)

    # Encode the prompt using the tokenizer
    prompt = "Hello GPT, how are you doing?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Generate the output
    output = model.generate(input_ids, top_k=10)
    
    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    main()
