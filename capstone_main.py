# Instamart Capstone


## Part C — Support Ticket Triage with Guardrails

The support-ticket triage pipeline processes tickets in their original file order.

### Privacy and input guardrails

- Empty tickets are skipped.
- Tickets exceeding 400 estimated input tokens are flagged as long.
- Email addresses, phone numbers and order IDs are redacted before model calls.
- Redaction substitutions: 278 total.
- Remaining matching personal information after redaction: 0.

### Prompt comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 11740 | 20632 |
| Tokens per classified ticket | 79.86 | 140.35 |

The few-shot prompt used 8892 additional tokens compared with zero-shot.
Its accuracy changed from 85.71% to
100.00%. The additional prompt cost is worthwhile
when the accuracy improvement justifies the extra token usage, particularly
for sensitive categories such as fraud.

### Budget demonstration

The zero-shot budget demonstration used a token budget of 6000.
It stopped before making further model calls for 76 tickets.

### Human review and rule validation

JSON and keyword classifications were accepted only after category validation.
Tickets with invalid model output, empty input, excessive length or exhausted
budget were routed to human review. Fraud classifications were routed to the
security team. Premium customers with other valid classifications were routed
to the premium desk; remaining valid tickets were routed to the support queue.

### Output files

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

The classification outputs were rule-validated; tickets flagged for review
were not automatically assigned a guessed category.


## Part D — Policy RAG Assistant and Refund Agent

### Policy RAG
- 40-word chunks with 10-word overlap: **15 chunks**.
- Answerable questions: **12/12** had the expected document in top 3.
- Expected facts matched in answers: **12/12**.
- Unanswerable questions correctly used the fallback: **2/2**.
- Zero-overlap fact failures: **2**; example: **What delivery time makes a completed order eligible for a full refund?**. With no overlap, a fact split across a chunk boundary may be separated from the terms needed to retrieve its continuation; the top three chunks may then omit the complete fact.
- Retrieval uses `TfidfVectorizer()` and cosine similarity. TF measures a term's frequency in a chunk; IDF gives greater weight to rarer terms across chunks. A limitation is that TF-IDF is lexical and may miss semantically equivalent wording.

### Refund agent scenario results
| ID | Status | Refund (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1045 |
| S2 | REFUND | 135.00 | 5 | 1045 |
| S3 | ESCALATE | 1999.00 | 5 | 1032 |
| S4 | DENIED | 0.00 | 4 | 717 |
| S5 | DENIED | 0.00 | 3 | 421 |
| S6 | NOT_FOUND | 0.00 | 2 | 130 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3168 |

### Safety and resumability demonstrations
- Token-limit demo: **FALLBACK_TOKEN_LIMIT** (438 tokens; requested tool not executed after limit breach).
- Timeout demo: **FALLBACK_TIMEOUT**.
- Loop demo: **FALLBACK_MAX_ITERATIONS** (6 model calls).
- Crash simulation: **SIMULATED_CRASH** after two tool observations.
- Resumed run: **REFUND**, final answer: `REFUND: INR 128.25 issued.`.
- Ordered tools across crash + resume: `get_order, search_policy, calculate_refund, issue_refund`.

### ReAct versus plan-and-execute (comparison)
The implemented ReAct agent chooses each next action after seeing the latest tool observation. In the normal S1 path it uses five model calls and four tool executions; the S2 path follows the same sequence, while S3 branches to human escalation after refund calculation. This observation-driven approach handles unexpected results naturally: a missing order returns `NOT_FOUND`, and a cancelled order returns `DENIED`, without executing a refund. A fixed plan-and-execute workflow could use fewer model calls by deciding the sequence in advance, but it is less adaptive if an order is missing, cancelled, or requires escalation. ReAct is selected here because safe refund processing depends on database and calculation observations before committing an action. The local mock model is deterministic, so measured calls/tokens are specific to this supplied test harness rather than a general production benchmark.
