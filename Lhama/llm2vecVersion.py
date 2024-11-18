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

documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = l2v.encode(documents)
print(d_reps)

q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
