import time

import backoff
import openai
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, LlamaTokenizer, T5ForConditionalGeneration, \
    LlamaForCausalLM,pipeline


class FlanT5:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.ckpt = "google/flan-t5-xl"
        self.ckpt = 'google/flan-t5-xxl'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.ckpt)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.eval()


    def generate(self, message):
        padded_sequence = self.tokenizer(
            message, padding=True, return_tensors="pt")
        input_ids = padded_sequence.input_ids.to(self.device)
        attention_mask = padded_sequence.attention_mask.to(self.device)

        generated_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=input_ids.shape[1] + 20,
            do_sample=True
        )
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return generated_text


class LLAMA:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = 'decapoda-research/llama-7b-hf'
        self.model = LlamaForCausalLM.from_pretrained(self.ckpt)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(self.ckpt)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    def generate(self, message):
        padded_sequence = self.tokenizer(
            message, padding=True, return_tensors="pt")
        input_ids = padded_sequence.input_ids.to(self.device)
        attention_mask = padded_sequence.attention_mask.to(self.device)


        generated_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=input_ids.shape[1] + 20,
            do_sample=True
        )
        generated_text = self.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        return generated_text

class ChatGLM2:
    def __init__(self):
        self.ckpt = "THUDM/chatglm2-6b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.ckpt, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()


    def generate(self,message):
        response,history = self.model.chat(self.tokenizer,message,history=[])
        return response


class LLAMA2:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.tokenizer.add_special_tokens({'pad_token': '[unk]'})
        self.model = pipeline(
        "text-generation",
        model=self.ckpt,
        torch_dtype=torch.float16,
        device_map="auto",
)
        self.init_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{}[/INST]"""


    def generate(self,message):
        message = self.init_prompt.format(message)
        padded_sequence = self.tokenizer(
            message, padding=True, return_tensors="pt")
        input_ids = padded_sequence.input_ids.to(self.device)

        response = self.model(message,
                              do_sample=True,
                              top_k=10,
                              num_return_sequences=1,
                              eos_token_id=self.tokenizer.eos_token_id,
                              max_length=input_ids.shape[1] + 50)

        return response[0]['generated_text'][len(message):].replace('\n','')


class ChatGPT:
    def __init__(self,name='generator'):
        self.api_key = []

        import random
        random.shuffle(self.api_key)
        self.name = name
        self.index = 0
        self.message = [{'role':'system','content':'You are a helpful assistant.'}]


    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def completion_with_backoff(self,**kwargs):
        openai.api_key = self.api_key[self.index]
        # print(openai.api_key)
        self.index = (self.index + 1) % len(self.api_key)
        ans = openai.ChatCompletion.create(**kwargs)
        return ans


    def create(self,messages):
        retry = True
        index = 0
        init_sleep_time = 5
        while retry:
            try:
                response = self.completion_with_backoff(
                    model='gpt-3.5-turbo',
                    messages=messages,
                )
                retry = False
            except Exception as e:
                print("503:{}".format(e))
                time.sleep(init_sleep_time)
                openai.api_key = self.api_key[index]
                # print(openai.api_key)
                index = (index + 1) % len(self.api_key)
                init_sleep_time += 1
        answer = response.choices[0].message['content']
        return answer


    def generate(self,message):
        if self.name == 'generator':
            messages = [{'role':'system','content':'You are a helpful assistant.'}]
            messages.append({'role':'user','content':message})
            return self.create(messages)
        else:
            self.message.append({'role':'user','content':message})
            ans = self.create(self.message)
            self.message.append({'role':"system",'content':ans})
            return ans
