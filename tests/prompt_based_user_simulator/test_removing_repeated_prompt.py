import json


def test_llama_generated_text_prompt_removal():

    generated_filepath = 'tests/generated_llama.json'
    generated = json.load(open(generated_filepath, 'r'))
    convs = list(generated.keys())
    replaced = 0
    for conv in convs:
        prompt_text = generated[conv]['prompted_text']
        gen_text = generated[conv]['generated_text']
        if gen_text.startswith(prompt_text):
            replaced += 1
            removed_prompt = gen_text.replace(prompt_text, '')
            assert len(removed_prompt) == len(gen_text) - len(prompt_text)
    assert replaced == 200


def test_t5_generated_text_prompt_removal_not_needed():

    generated_filepath = 'tests/generated_flant5.json'
    generated = json.load(open(generated_filepath, 'r'))
    convs = list(generated.keys())
    replaced = 0
    for conv in convs:
        prompt_text = generated[conv]['prompted_text']
        gen_text = generated[conv]['generated_text']
        if gen_text.startswith(prompt_text):
            replaced += 1
            removed_prompt = gen_text.replace(prompt_text, '')
            assert len(removed_prompt) == len(gen_text) - len(prompt_text)
    assert replaced == 0
