import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

sstart=time.time()
 
####### Section 1. Set up #######
torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct" # please replace with local model path
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
 
####### Section 2. GPU Warm up #######
messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

input_text = ""
for message in messages:
    input_text += f"{message['role']}: {message['content']}\n"

#input_text="hello, sample text"

input_ids_tensor = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
position_ids = torch.arange(input_ids_tensor.size(1), dtype=torch.long, device=input_ids_tensor.device)
position_ids = position_ids.unsqueeze(0).expand(input_ids_tensor.size(0), -1)

print("input_ids_tensor : ",input_ids_tensor)

with torch.no_grad():
    x = model.model.embed_tokens(input_ids_tensor)
    print("임베딩 출력:", x)

    seq_length = input_ids_tensor.size(1)  # 입력 시퀀스의 길이
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids_tensor.device)
    position_ids = position_ids.unsqueeze(0).expand(input_ids_tensor.size(0), -1)

    attention_mask = torch.ones(input_ids_tensor.shape, device=input_ids_tensor.device)

    for layer in model.model.layers:
        x = layer(x, position_ids=position_ids)
        x = x[0]  # 첫 번째 튜플 요소를 가져옴 (hidden states)

    print("after layers:",x)
    #print("\n\n\n",dir(model),"\n\n\n")
    print("m.m.attn:",model.model._attn_implementation)
    print("m.autoset:",model._autoset_attn_implementation)
    print("m.m.autoset:",model.model._autoset_attn_implementation)
    x = model.model._autoset_attn_implementation(x)
    print("after autoset implementation:",x)
    x = model.model.norm(x)
    print("after norm:",x)
    logits = model.lm_head(x)
    print("after lm_head:",logits)

    predicted_token_ids = torch.argmax(logits, dim=-1)
    decoded_output = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
    #decoded_output = tokenizer.decode(logits, skip_special_tokens=True)
print("출력:",decoded_output)


#output = pipe(messages, **generation_args)
#print(output[0]['generated_text'])


"""

####### Section 3. Load data and Inference -> Performance evaluation part #######
start = time.time()
data = load_dataset("json", data_files="test_dataset.jsonl")['train']
outs = pipe(KeyDataset(data, 'message'), batch_size=16, **generation_args)
end = time.time()
 
####### Section 4. Accuracy (Just for leasderboard) #######
print("===== Answers =====")
correct = 0
for i, out in enumerate(outs):
    correct_answer = data[i]["answer"]
    answer = out[0]["generated_text"].lstrip().replace("\n","")
    if answer == correct_answer:
        correct += 1
    print(answer)
    
eend=time.time()
 
print("===== Perf result =====")
print("Total_time: ",eend-sstart)
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")

"""
