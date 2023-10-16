import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Note: tried
# from transformers import AutoTokenizer, AutoModelForCausalLM
# TOKENIZER = AutoTokenizer.from_pretrained("decapoda-research/llama-13b-hf")
# get an error:
# The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization.  # noqa
# The tokenizer class you load from this checkpoint is 'LLaMATokenizer'.
# The class this function is called from is 'LlamaTokenizer'.

# https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaModel  # noqa
TOKENIZER = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
MODEL = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)


def test_text_generation():

    prompt = open('tests/sample_prompt.txt', 'r').read()

    padded_sequence = TOKENIZER(
        prompt,
        padding=True,
        return_tensors='pt'
    )

    input_ids = padded_sequence.input_ids.to(DEVICE)
    attention_mask = padded_sequence.attention_mask.to(DEVICE)
    generated_ids = MODEL.generate(
        input_ids,
        attention_mask=attention_mask,
        temperature=0.8,
        pad_token_id=TOKENIZER.pad_token_id,
        max_length=input_ids.shape[1]+30,
        do_sample=True)
    generated_text = TOKENIZER.decode(
        generated_ids[0], skip_special_tokens=True)
    print(generated_text)
    assert generated_text
