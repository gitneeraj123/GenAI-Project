from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
        "return_full_text": False
    }
)

prompt = "<|user|>\nWhat is the capital of France?\n<|assistant|>"

result = llm.invoke(prompt)

print(result)