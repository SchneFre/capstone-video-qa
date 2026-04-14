import time
from langchain.chat_models import ChatOpenAI

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

    Evaluate the model answer based on:
    - Correctness
    - Completeness
    - Relevance

    Give:
    1. A score from 1 to 10
    2. A short explanation

    Format:
    Score: <number>
    Explanation: <text>
    """

    start = time.perf_counter()
    response = judge.predict(prompt)
    latency = time.perf_counter() - start

    return response, latency


def inspect_retrieval(vector_db, question):
    start = time.perf_counter()
    docs = vector_db.similarity_search(question, k=3)
    latency = time.perf_counter() - start

    return [doc.page_content for doc in docs], latency


def run_full_evaluation(qa_chain, vector_db, openai_api_key):

    evaluation_data = [
        {
            "question": "How many aircrafts have been shot down?",
            "ground_truth": "Two U.S. aircraft have been shot down in Iran. A total of two aircraft have been reported to have been downed."
        },
        {
            "question": "what happened to the pilots?",
            "ground_truth": "One pilot was rescued and the other is still missing"
        },
        {
            "question": "What did trump do?",
            "ground_truth": "President Trump is requesting a record $1.5 trillion dollar defense budget from US taxpayers."
        }
    ]

    results = []

    for item in evaluation_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # 🔹 Retrieval latency + docs
        retrieved_docs, retrieval_latency = inspect_retrieval(vector_db, question)

        # 🔹 QA latency
        start = time.perf_counter()
        prediction = qa_chain.run(question)
        qa_latency = time.perf_counter() - start

        # 🔹 Evaluation latency
        evaluation, eval_latency = evaluate_answer(
            question,
            prediction,
            ground_truth,
            openai_api_key
        )

        results.append({
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "retrieved_docs": retrieved_docs,
            "retrieval_latency": retrieval_latency,
            "qa_latency": qa_latency,
            "evaluation": evaluation,
            "eval_latency": eval_latency
        })

    return results