from helper_lib.assignment5_llm.llm_base import load_llm, load_tokenizer, generate_answer

tokenizer = load_tokenizer()
model = load_llm(use_finetuned=True)

text = generate_answer(
    model,
    tokenizer,
    "What is deep learning?",
    max_new_tokens=60
)

print(text)