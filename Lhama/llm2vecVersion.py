import torch
from huggingface_hub import login
from llm2vec.llm2vec import LLM2Vec

login("")

l2v = LLM2Vec.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

l2v.save("Llama-3.1-8B-Emb")

instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)

queries = [
    [instruction, "How much protein should a female eat"],
    [instruction, "summit define"],
]

q_reps = l2v.encode(queries)

print("Query representations generated successfully!")
