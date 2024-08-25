from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    model_id = "microsoft/Phi-3-medium-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
    for nmod, _ in model.named_modules():
        print(nmod)

    print("-" * 30)
    for nmod, _ in model.named_children():
        print(nmod)
