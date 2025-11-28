# helper_lib/assignment5_llm/llm_base.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained("./finetuned_gpt2")
    tok.pad_token = tok.eos_token
    return tok

def load_llm(use_finetuned=True):
    if use_finetuned:
        model = AutoModelForCausalLM.from_pretrained("./finetuned_gpt2")
    else:
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    # FIX: GPT-2 has no pad token → assign eos as pad
    model.config.pad_token_id = model.config.eos_token_id
    return model

def generate_answer(model, tokenizer, question, max_new_tokens=80):
    model.eval()

    prompt = (
        f"Question: {question}\n"
        "That is a great question. The answer is: "
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure ending sentence exists
    if "Let me know if you have any other questions." not in text:
        text += " Let me know if you have any other questions."

    return text