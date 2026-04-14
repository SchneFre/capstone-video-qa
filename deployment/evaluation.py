
import time
import psutil
import os
import tiktoken
from langchain.chat_models import ChatOpenAI

# ---------------------------
# COST + TOKEN ESTIMATION
# ---------------------------
def estimate_cost(text, model="gpt-4o"):
    """
    Rough cost estimation using token count.
    Update pricing if needed.
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = len(enc.encode(text))
    # Approx GPT-4o pricing (update if needed)
    input_cost_per_1k = 0.005
    output_cost_per_1k = 0.015
    return tokens, (tokens / 1000) * ((input_cost_per_1k + output_cost_per_1k) / 2)

# ---------------------------
# MEMORY USAGE
# ---------------------------
def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# ---------------------------
# EVALUATION MODEL
# ---------------------------
def evaluate_answer(question, prediction, ground_truth, openai_api_key):
    judge = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=openai_api_key
    )
    prompt = f"""
    You are an expert evaluator for a QA system.
    Question: {question}
    Ground Truth Answer: {ground_truth}
    Model Answer: {prediction}
    Evaluate based on:
    - Correctness
    - Completeness
    - Relevance
    Give:
    Score: 1-10
    Explanation: short reasoning
    """
    start_mem = get_memory_mb()
    start = time.perf_counter()
    response = judge.predict(prompt)
    latency = time.perf_counter() - start
    end_mem = get_memory_mb()
    mem_used = end_mem - start_mem
    tokens, cost = estimate_cost(prompt + response)
    return response, latency, mem_used, tokens, cost

# ---------------------------
# RETRIEVAL INSPECTION
# ---------------------------
def inspect_retrieval(vector_db, question):
    start_mem = get_memory_mb()
    start = time.perf_counter()
    docs = vector_db.similarity_search(question, k=3)
    latency = time.perf_counter() - start
    end_mem = get_memory_mb()
    mem_used = end_mem - start_mem
    return [doc.page_content for doc in docs], latency, mem_used
def print_results(results):
    for r in results:
        print("\n==============================")
        print("Question:", r["question"])
        print("QA Latency:", r["qa_latency_sec"], "sec")
        print("Retrieval Latency:", r["retrieval_latency_sec"], "sec")
        print("Eval Latency:", r["eval_latency_sec"], "sec")
        print("Memory (QA):", r["qa_memory_mb"], "MB")
        print("Tokens:", r["tokens_estimated"])
        print("Cost ($):", r["cost_estimated_usd"])
        print("Evaluation:", r["evaluation"])

# ---------------------------
# FULL EVALUATION PIPELINE
# ---------------------------
def run_full_evaluation(qa_chain, vector_db, openai_api_key):
    evaluation_data = [
        {
            "question": "How many aircrafts have been shot down?",
            "ground_truth": "Two U.S. aircraft have been shot down in Iran."
        },
        {
            "question": "what happened to the pilots?",
            "ground_truth": "One pilot was rescued and the other is missing"
        },
        {
            "question": "What did trump do?",
            "ground_truth": "Requested a $1.5 trillion defense budget."
        }
    ]
    results = []
    for item in evaluation_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        # Retrieval
        retrieved_docs, retr_latency, retr_mem = inspect_retrieval(vector_db, question)
        # QA
        start_mem = get_memory_mb()
        start = time.perf_counter()
        prediction = qa_chain.run(question)
        qa_latency = time.perf_counter() - start
        qa_mem = get_memory_mb() - start_mem
        # Evaluation
        evaluation, eval_latency, eval_mem, tokens, cost = evaluate_answer(
            question,
            prediction,
            ground_truth,
            openai_api_key
        )
        results.append({
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            # retrieval metrics
            "retrieved_docs": retrieved_docs,
            "retrieval_latency_sec": retr_latency,
            "retrieval_memory_mb": retr_mem,
            # QA metrics
            "qa_latency_sec": qa_latency,
            "qa_memory_mb": qa_mem,
            # evaluation metrics
            "evaluation": evaluation,
            "eval_latency_sec": eval_latency,
            "eval_memory_mb": eval_mem,
            # cost metrics
            "tokens_estimated": tokens,
            "cost_estimated_usd": cost
        })
    return results
