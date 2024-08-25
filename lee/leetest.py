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


##padding 문제!!!!!

max_length = 128

input_ids = [tokenizer(message["content"], return_tensors="pt",padding="max_length", truncation=True,max_length=max_length)["input_ids"] for message in messages]


input_ids_tensor = torch.cat(input_ids, dim=0)  # dim=0: 배치 차원에서 텐서들을 합침
print("input_ids_tensor : ",input_ids_tensor)

position_ids = torch.arange(0, max_length, dtype=torch.long, device=input_ids_tensor.device).unsqueeze(0).expand(input_ids_tensor.size(0), -1)


with torch.no_grad():
    x = model.model.embed_tokens(input_ids_tensor)
    x = model.model.embed_dropout(x)

    for i in range(40):
        x = model.model.layers[i](x, position_ids=position_ids)
        x = x[0]

    x = model.model.norm(x)
    x = model.lm_head(x)

    predicted_token_ids = torch.argmax(x, dim=-1)
    print("predicted_token_ids : ",predicted_token_ids)
    decoded_output = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

print(decoded_output)


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
