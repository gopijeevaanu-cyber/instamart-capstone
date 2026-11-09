"""
Part D - Instamart Policy RAG Assistant and Refund Agent
Run from the capstone project root:
    py part_d_agent.py

Uses only the supplied offline mock_llm.py. No network/API key required.
"""
import os
import re
import json
import time
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
REFUND_LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
README_FILE = ROOT / "README.md"


# ---------------------------------------------------------------------
# 1. Policy chunking and TF-IDF retrieval
# ---------------------------------------------------------------------
def load_policy_chunks(chunk_size=40, overlap=10):
    """Create word chunks with source metadata. Step = chunk_size - overlap."""
    chunks = []
    step = chunk_size - overlap
    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        if not words:
            continue
        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]
            if not chunk_words:
                break
            chunks.append({
                "source": path.name,
                "score": 0.0,
                "text": " ".join(chunk_words),
            })
            if start + chunk_size >= len(words):
                break
    return chunks


class PolicyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def retrieve(self, query, k=3):
        if self.matrix is None:
            return []
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"],
            }
            for i in ranked
        ]


POLICY_CHUNKS = load_policy_chunks(40, 10)
RETRIEVER = PolicyRetriever(POLICY_CHUNKS)


def retrieve(query, k=3):
    return RETRIEVER.retrieve(query, k=k)


def answer(question):
    results = retrieve(question, k=3)
    context_parts = []
    for item in results:
        context_parts.append(f"[{item['source']}]\n{item['text']}")
    context = "\n".join(context_parts)
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer using only the supplied policy context. "
                "If the answer is not present in the context, reply exactly: "
                "I could not find that information in the policy documents."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
        },
    ]
    response = mock_llm.chat(messages)
    return {
        "answer": response.choices[0].message.content,
        "sources": list(dict.fromkeys(x["source"] for x in results)),
        "retrieved": results,
    }


def evaluate_policy():
    evaluation = pd.read_csv(RAW_DIR / "policy_eval.csv")
    rows = []
    for _, row in evaluation.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"])
        expected_fact = "" if pd.isna(row["expected_fact"]) else str(row["expected_fact"])
        result = answer(question)
        is_unanswerable = expected_doc.strip().upper() == "NONE"
        rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": result["answer"],
            "sources": ";".join(result["sources"]),
            "expected_doc_in_top3": (
                is_unanswerable or expected_doc in result["sources"]
            ),
            "expected_fact_in_answer": (
                False if is_unanswerable
                else expected_fact in result["answer"]
            ),
            "unanswerable_fallback": (
                is_unanswerable
                and result["answer"].strip()
                == "I could not find that information in the policy documents."
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "policy_evaluation_results.csv", index=False)

    answerable = df[df["expected_doc"] != "NONE"]
    unanswerable = df[df["expected_doc"] == "NONE"]
    summary = {
        "chunks_with_overlap": len(POLICY_CHUNKS),
        "answerable_questions": len(answerable),
        "expected_doc_top3": int(answerable["expected_doc_in_top3"].sum()),
        "expected_fact_matches": int(answerable["expected_fact_in_answer"].sum()),
        "unanswerable_questions": len(unanswerable),
        "unanswerable_fallbacks": int(unanswerable["unanswerable_fallback"].sum()),
    }

    # Repeat retrieval/evaluation with zero overlap.
    zero_chunks = load_policy_chunks(40, 0)
    zero_retriever = PolicyRetriever(zero_chunks)
    zero_rows = []
    for _, row in evaluation.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"])
        expected_fact = "" if pd.isna(row["expected_fact"]) else str(row["expected_fact"])
        results = zero_retriever.retrieve(question, 3)
        context = "\n".join(f"[{x['source']}]\n{x['text']}" for x in results)
        messages = [
            {"role": "system", "content":
             "POLICY ASSISTANT. Answer using only the supplied policy context. "
             "If the answer is not present in the context, reply exactly: "
             "I could not find that information in the policy documents."},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
        ]
        response = mock_llm.chat(messages).choices[0].message.content
        is_unanswerable = expected_doc.strip().upper() == "NONE"
        zero_rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": response,
            "sources": ";".join(dict.fromkeys(x["source"] for x in results)),
            "expected_doc_in_top3": is_unanswerable or expected_doc in [x["source"] for x in results],
            "expected_fact_in_answer": False if is_unanswerable else expected_fact in response,
        })
    zero_df = pd.DataFrame(zero_rows)
    zero_df.to_csv(OUTPUT_DIR / "policy_evaluation_zero_overlap.csv", index=False)
    zero_answerable = zero_df[zero_df["expected_doc"] != "NONE"]
    zero_failures = zero_answerable[~zero_answerable["expected_fact_in_answer"]]
    summary["zero_overlap_fact_failures"] = int(len(zero_failures))
    summary["zero_overlap_failure_question"] = (
        str(zero_failures.iloc[0]["question"]) if len(zero_failures) else "None observed"
    )

    print("\n========== POLICY RAG ==========")
    print("40-word chunks, 10-word overlap:", len(POLICY_CHUNKS))
    print("Query: refund policy late delivery")
    for item in retrieve("refund policy late delivery", 3):
        print(f"  {item['source']} | score={item['score']:.4f}")
    print("Evaluation:", summary)
    print("Zero-overlap fact failures:", len(zero_failures))
    if len(zero_failures):
        print("Example zero-overlap failure:", zero_failures.iloc[0]["question"])
    else:
        print("No zero-overlap fact failure was observed in this run; inspect "
              "policy_evaluation_zero_overlap.csv and report this honestly.")
    print("Saved policy_evaluation_results.csv and policy_evaluation_zero_overlap.csv")
    return summary, zero_df


# ---------------------------------------------------------------------
# 2. Tool implementations and tool specifications
# ---------------------------------------------------------------------
def get_order(order_id):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            """SELECT order_id, status, order_amount, delivery_minutes
               FROM orders WHERE order_id = ?""",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found", "order_id": order_id}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": None if row[3] is None else float(row[3]),
    }


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount) if delivery_minutes is not None and float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(amount, 2)}


def append_ledger(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with REFUND_LEDGER_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def issue_refund(order_id, refund_inr):
    entry = append_ledger("issue_refund", order_id, refund_inr)
    return {"order_id": order_id, "refund_inr": round(float(refund_inr), 2),
            "ledger_timestamp": entry["timestamp"], "status": "issued"}


def escalate_to_human(order_id, refund_inr):
    entry = append_ledger("escalate_to_human", order_id, refund_inr)
    return {"order_id": order_id, "refund_inr": round(float(refund_inr), 2),
            "ledger_timestamp": entry["timestamp"], "status": "escalated"}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order by its ID in the Instamart SQLite database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the three most relevant policy chunks for a query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Return the full order amount if delivery took more than 45 minutes, otherwise zero.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an approved refund and append an audit entry to the JSONL refund ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead and append an audit entry.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]


# ---------------------------------------------------------------------
# 3. ReAct agent, safety limits, checkpoints and resume
# ---------------------------------------------------------------------
def _jsonable_tool_calls(tool_calls):
    result = []
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None)
        if fn is not None:
            result.append({
                "id": getattr(tc, "id", ""),
                "name": getattr(fn, "name", ""),
                "arguments": getattr(fn, "arguments", {}),
            })
        elif isinstance(tc, dict):
            result.append(tc)
        else:
            result.append({
                "id": getattr(tc, "id", ""),
                "name": getattr(tc, "name", ""),
                "arguments": getattr(tc, "arguments", {}),
            })
    return result


def _response_parts(response):
    choice = response.choices[0]
    message = choice.message
    return message.content, _jsonable_tool_calls(getattr(message, "tool_calls", [])), response.usage.total_tokens


def _checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(run_id, scenario_id, messages, tokens_used, step, tools_executed, started_at):
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step": step,
        "tools_executed": tools_executed,
        "elapsed_seconds": time.monotonic() - started_at,
    }
    _checkpoint_path(run_id).write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def _fallback(status, steps, llm_calls, tokens_used, detail):
    return {
        "status": status,
        "final_answer": f"{status}: {detail}",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def _continue_agent(run_id, max_iterations, token_limit, time_limit_seconds,
                    started_at=None, crash_after_tools=None):
    checkpoint_file = _checkpoint_path(run_id)
    state = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    scenario_id = state["scenario_id"]
    messages = state["messages"]
    tokens_used = int(state["tokens_used"])
    tools_executed = state["tools_executed"]
    steps = state.get("steps", [])
    llm_calls = int(state.get("llm_calls", 0))
    started_at = time.monotonic() if started_at is None else started_at
    # Account for time spent before a resume as part of the same run.
    prior_elapsed = float(state.get("elapsed_seconds", 0.0))
    start_clock = time.monotonic() - prior_elapsed

    while llm_calls < max_iterations:
        elapsed = time.monotonic() - start_clock
        if elapsed >= time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT", steps, llm_calls, tokens_used,
                             "time limit reached; handing the case to a human.")
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        content, tool_calls, call_tokens = _response_parts(response)
        tokens_used += int(call_tokens)

        # Stop immediately after the model call if the cumulative total is over budget.
        if tokens_used > token_limit:
            _save_checkpoint(run_id, scenario_id, messages, tokens_used,
                             len(steps), tools_executed, start_clock)
            state = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            state["steps"] = steps
            state["llm_calls"] = llm_calls
            state["elapsed_seconds"] = time.monotonic() - start_clock
            checkpoint_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
            return _fallback("FALLBACK_TOKEN_LIMIT", steps, llm_calls, tokens_used,
                             "token budget exceeded before executing the requested tool; handing the case to a human.")

        if not tool_calls:
            final = content or "FALLBACK: the assistant returned no final answer."
            status = final.split(":", 1)[0].strip().upper() if ":" in final else "FALLBACK"
            state.update({"messages": messages, "tokens_used": tokens_used,
                          "step": len(steps), "tools_executed": tools_executed,
                          "steps": steps, "llm_calls": llm_calls,
                          "elapsed_seconds": time.monotonic() - start_clock})
            checkpoint_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"status": status, "final_answer": final, "steps": steps,
                    "llm_calls": llm_calls, "tokens_used": tokens_used}

        # Record assistant tool-call message, then execute each requested tool.
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        messages.append(assistant_message)
        for call in tool_calls:
            name = call["name"]
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                try:
                    observation = TOOL_MAP[name](**args)
                except Exception as exc:
                    observation = {"error": f"{type(exc).__name__}: {exc}"}
            tools_executed.append(name)
            step = {"tool": name, "args": args, "observation": observation}
            steps.append(step)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(observation, ensure_ascii=False),
            })
            # Save immediately after every tool observation.
            _save_checkpoint(run_id, scenario_id, messages, tokens_used,
                             len(steps), tools_executed, start_clock)
            state = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            state["steps"] = steps
            state["llm_calls"] = llm_calls
            state["elapsed_seconds"] = time.monotonic() - start_clock
            checkpoint_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            if crash_after_tools is not None and len(tools_executed) == crash_after_tools:
                return {
                    "status": "SIMULATED_CRASH",
                    "final_answer": "SIMULATED_CRASH: checkpoint saved; resume to continue.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                    "run_id": run_id,
                }

    return _fallback("FALLBACK_MAX_ITERATIONS", steps, llm_calls, tokens_used,
                     "maximum model iterations reached; handing the case to a human.")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:10]}"
    messages = [
        {"role": "system", "content":
         "AGENT. You are an Instamart refund agent. Use the available tools to check the order, "
         "retrieve policy, calculate the eligible refund, and issue or escalate it. "
         "Do not invent order information."},
        {"role": "user", "content": user_message},
    ]
    started = time.monotonic()
    _save_checkpoint(run_id, scenario_id, messages, 0, 0, [], started)
    # Include initialized state fields so resume can restore the full run.
    state = json.loads(_checkpoint_path(run_id).read_text(encoding="utf-8"))
    state.update({"steps": [], "llm_calls": 0, "elapsed_seconds": 0.0})
    _checkpoint_path(run_id).write_text(json.dumps(state, indent=2), encoding="utf-8")
    return _continue_agent(run_id, max_iterations, token_limit,
                           time_limit_seconds, started_at=started,
                           crash_after_tools=crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    if not _checkpoint_path(run_id).exists():
        raise FileNotFoundError(f"No checkpoint found for run_id={run_id}")
    return _continue_agent(run_id, max_iterations, token_limit, time_limit_seconds)


def append_jsonl(path, item):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# 4. Scenario batch, safety demonstrations, crash/resume demonstration
# ---------------------------------------------------------------------
def run_scenarios():
    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv")
    results = []
    for _, row in scenarios.iterrows():
        scenario_id = str(row["scenario_id"])
        result = run_agent(
            str(row["message"]),
            scenario_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            run_id=f"scenario_{scenario_id}_{uuid.uuid4().hex[:8]}",
        )
        expected = str(row["expected_outcome"])
        expected_refund = float(row["expected_refund_inr"])
        actual_refund = 0.0
        for step in result["steps"]:
            obs = step["observation"]
            if isinstance(obs, dict) and "refund_inr" in obs:
                actual_refund = float(obs["refund_inr"])
        record = {
            "scenario_id": scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }
        append_jsonl(AGENT_RUNS_FILE, record)
        results.append({
            "scenario_id": scenario_id,
            "expected_outcome": expected,
            "status": result["status"],
            "expected_refund_inr": expected_refund,
            "refund_inr": actual_refund,
            "model_calls": result["llm_calls"],
            "tokens": result["tokens_used"],
            "result_match": (
                (expected == "FALLBACK" and result["status"].startswith("FALLBACK"))
                or (expected == result["status"])
            ),
            "refund_match": abs(expected_refund - actual_refund) <= 0.01,
        })
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "agent_scenario_results.csv", index=False)
    return df


def run_safety_demos():
    s1 = "My order ORD-100048 reached me very late. I would like a refund."
    token_demo = run_agent(s1, "DEMO_TOKEN", token_limit=200, run_id="demo_token_limit")
    timeout_demo = run_agent(s1, "DEMO_TIMEOUT", time_limit_seconds=0, run_id="demo_timeout")
    loop_demo = run_agent("[LOOP] Refund order ORD-100048.", "DEMO_LOOP",
                          max_iterations=6, run_id="demo_max_iterations")
    return {
        "token_limit": token_demo,
        "timeout": timeout_demo,
        "max_iterations": loop_demo,
    }


def run_resume_demo():
    message = "My order ORD-100048 reached me very late. I would like a refund."
    run_id = "demo_resume_s1"
    # Remove an old checkpoint so this demonstration is repeatable.
    cp = _checkpoint_path(run_id)
    if cp.exists():
        cp.unlink()
    crashed = run_agent(message, "DEMO_RESUME_S1", run_id=run_id,
                        max_iterations=6, crash_after_tools=2)
    resumed = resume_agent(run_id, max_iterations=6)
    return crashed, resumed


# ---------------------------------------------------------------------
# 5. README report
# ---------------------------------------------------------------------
def update_readme(policy_summary, scenario_df, safety, crash, resumed):
    scenario_lines = [
        "| ID | Status | Refund (INR) | Model calls | Tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in scenario_df.iterrows():
        scenario_lines.append(
            f"| {r['scenario_id']} | {r['status']} | {r['refund_inr']:.2f} | "
            f"{int(r['model_calls'])} | {int(r['tokens'])} |"
        )

    tool_order = [x["tool"] for x in resumed["steps"]]
    safety_lines = [
        f"- Token-limit demo: **{safety['token_limit']['status']}** "
        f"({safety['token_limit']['tokens_used']} tokens; requested tool not executed after limit breach).",
        f"- Timeout demo: **{safety['timeout']['status']}**.",
        f"- Loop demo: **{safety['max_iterations']['status']}** "
        f"({safety['max_iterations']['llm_calls']} model calls).",
        f"- Crash simulation: **{crash['status']}** after two tool observations.",
        f"- Resumed run: **{resumed['status']}**, final answer: `{resumed['final_answer']}`.",
        f"- Ordered tools across crash + resume: `{', '.join(tool_order)}`.",
    ]

    section = f"""

## Part D — Policy RAG Assistant and Refund Agent

### Policy RAG
- 40-word chunks with 10-word overlap: **{policy_summary['chunks_with_overlap']} chunks**.
- Answerable questions: **{policy_summary['expected_doc_top3']}/{policy_summary['answerable_questions']}** had the expected document in top 3.
- Expected facts matched in answers: **{policy_summary['expected_fact_matches']}/{policy_summary['answerable_questions']}**.
- Unanswerable questions correctly used the fallback: **{policy_summary['unanswerable_fallbacks']}/{policy_summary['unanswerable_questions']}**.
- Zero-overlap fact failures: **{policy_summary['zero_overlap_fact_failures']}**; example: **{policy_summary['zero_overlap_failure_question']}**. With no overlap, a fact split across a chunk boundary may be separated from the terms needed to retrieve its continuation; the top three chunks may then omit the complete fact.
- Retrieval uses `TfidfVectorizer()` and cosine similarity. TF measures a term's frequency in a chunk; IDF gives greater weight to rarer terms across chunks. A limitation is that TF-IDF is lexical and may miss semantically equivalent wording.

### Refund agent scenario results
{chr(10).join(scenario_lines)}

### Safety and resumability demonstrations
{chr(10).join(safety_lines)}

### ReAct versus plan-and-execute (comparison)
The implemented ReAct agent chooses each next action after seeing the latest tool observation. In the normal S1 path it uses five model calls and four tool executions; the S2 path follows the same sequence, while S3 branches to human escalation after refund calculation. This observation-driven approach handles unexpected results naturally: a missing order returns `NOT_FOUND`, and a cancelled order returns `DENIED`, without executing a refund. A fixed plan-and-execute workflow could use fewer model calls by deciding the sequence in advance, but it is less adaptive if an order is missing, cancelled, or requires escalation. ReAct is selected here because safe refund processing depends on database and calculation observations before committing an action. The local mock model is deterministic, so measured calls/tokens are specific to this supplied test harness rather than a general production benchmark.
"""
    existing = README_FILE.read_text(encoding="utf-8") if README_FILE.exists() else ""
    marker = "## Part D — Policy RAG Assistant and Refund Agent"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()
    README_FILE.write_text(existing + section, encoding="utf-8")


def main():
    print("Using database:", DB_PATH)
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Expected database not found: {DB_PATH}")
    if not (RAW_DIR / "agent_scenarios.csv").exists():
        raise FileNotFoundError("Missing data/raw/agent_scenarios.csv")
    if not list(POLICY_DIR.glob("*.txt")):
        raise FileNotFoundError("No policy .txt files found in data/policies")

    print("\nPolicy chunk counts by source:")
    print(pd.Series([x["source"] for x in POLICY_CHUNKS]).value_counts().to_string())

    policy_summary, _ = evaluate_policy()

    # Clear this run's two JSONL output files to avoid duplicate rows on reruns.
    for path in (AGENT_RUNS_FILE, REFUND_LEDGER_FILE):
        if path.exists():
            path.unlink()

    print("\n========== SEVEN AGENT SCENARIOS ==========")
    scenario_df = run_scenarios()
    print(scenario_df.to_string(index=False))

    print("\n========== SAFETY DEMOS ==========")
    safety = run_safety_demos()
    for name, result in safety.items():
        print(name, "=>", result["status"],
              "| calls:", result["llm_calls"],
              "| tokens:", result["tokens_used"])

    print("\n========== CRASH / RESUME DEMO ==========")
    crash, resumed = run_resume_demo()
    print("Before resume:", crash["status"],
          "| tools:", [x["tool"] for x in crash["steps"]])
    print("After resume:", resumed["status"],
          "| tools:", [x["tool"] for x in resumed["steps"]])
    print("Final:", resumed["final_answer"])

    update_readme(policy_summary, scenario_df, safety, crash, resumed)

    print("\n========== OUTPUTS CREATED ==========")
    for path in [
        OUTPUT_DIR / "policy_evaluation_results.csv",
        OUTPUT_DIR / "policy_evaluation_zero_overlap.csv",
        OUTPUT_DIR / "agent_scenario_results.csv",
        AGENT_RUNS_FILE,
        REFUND_LEDGER_FILE,
        STATE_DIR,
        README_FILE,
    ]:
        print(path.relative_to(ROOT), "FOUND" if path.exists() else "MISSING")
    print("\nPart D run complete.")


if __name__ == "__main__":
    main()
