import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TOKENIZER = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
MODEL = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
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
