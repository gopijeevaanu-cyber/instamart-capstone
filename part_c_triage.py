import re
import json
import time
import random
from pathlib import Path

import pandas as pd
import mock_llm


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

TICKETS_FILE = RAW_DIR / "tickets.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

ALLOWED_CATEGORIES = {
    "billing", "delivery", "technical", "fraud"
}


# --------------------------------------------------
# PRIVACY: EXACT REQUIRED REDACTION PATTERNS
# --------------------------------------------------

REDACTION_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")
]


def redact_text(text):
    text = "" if pd.isna(text) else str(text)
    counts = {}

    for name, pattern, placeholder in REDACTION_PATTERNS:
        text, count = pattern.subn(placeholder, text)
        counts[name] = count

    return text, counts


def check_redaction(text):
    for _, pattern, _ in REDACTION_PATTERNS:
        assert pattern.search(text) is None, (
            "Privacy check failed: personal information remains!"
        )


# --------------------------------------------------
# SYSTEM PROMPTS
# --------------------------------------------------

ZERO_SHOT_PROMPT = """
CLASSIFY the support ticket into exactly one category.

Allowed categories:
billing
delivery
technical
fraud

Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}
Do not include markdown or any text outside the JSON.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the support ticket into exactly one category.

Allowed categories:
billing
delivery
technical
fraud

EXAMPLES:
Ticket: I was charged twice for my order.
Answer: {"category": "billing", "reason": "Duplicate charge"}

Ticket: Someone accessed my account without permission.
Answer: {"category": "fraud", "reason": "Unauthorized account access"}

Ticket: My app keeps crashing when I open it.
Answer: {"category": "technical", "reason": "App crash"}

Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}
Do not include markdown or any text outside the JSON.
"""


# --------------------------------------------------
# RESPONSE VALIDATION
# --------------------------------------------------

def validate_response(content):
    """
    Returns category and status.
    Valid JSON with an invalid category is flagged.
    Keyword extraction is only used when JSON parsing fails.
    """

    try:
        data = json.loads(content)

        if (
            isinstance(data, dict)
            and data.get("category") in ALLOWED_CATEGORIES
        ):
            return data["category"], "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())

        found = [
            word for word in words
            if word in ALLOWED_CATEGORIES
        ]

        if len(found) == 1:
            return found[0], "classified_keyword"

        return "", "flagged_invalid_output"


# --------------------------------------------------
# RETRY FUNCTION
# --------------------------------------------------

def call_with_retry(messages, ticket_id):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if attempt >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY | ticket={ticket_id} "
                f"| retry_number={attempt} "
                f"| wait={wait:.4f}s"
            )

            time.sleep(wait)

    raise RuntimeError(f"Retries exhausted for ticket {ticket_id}")


# --------------------------------------------------
# LOAD TICKETS AND CUSTOMER SEGMENTS
# --------------------------------------------------

tickets = pd.read_csv(TICKETS_FILE)
segments = pd.read_csv(SEGMENTS_FILE)

required_ticket_columns = {
    "ticket_id", "customer_id", "order_id", "text", "true_category"
}

missing_columns = required_ticket_columns - set(tickets.columns)

if missing_columns:
    raise ValueError(
        f"Missing ticket columns: {sorted(missing_columns)}"
    )

# Create customer lookup.
# If is_premium does not exist, customers are treated as not premium.
if "is_premium" not in segments.columns:
    segments["is_premium"] = False

segment_column = (
    "segment_name"
    if "segment_name" in segments.columns
    else "segment"
    if "segment" in segments.columns
    else None
)

if segment_column is None:
    segments["segment_name"] = ""
    segment_column = "segment_name"


def is_true(value):
    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


customer_lookup = {}

for _, row in segments.iterrows():
    customer_lookup[str(row["customer_id"])] = {
        "segment_name": (
            str(row[segment_column])
            if pd.notna(row[segment_column])
            else ""
        ),
        "is_premium": is_true(row["is_premium"])
    }


# --------------------------------------------------
# REDACT ALL TICKETS BEFORE ANY MODEL CALL
# --------------------------------------------------

redacted_texts = []
redaction_totals = {
    "email": 0,
    "phone": 0,
    "order_id": 0
}

for original_text in tickets["text"]:
    safe_text, counts = redact_text(original_text)
    check_redaction(safe_text)

    redacted_texts.append(safe_text)

    for key in redaction_totals:
        redaction_totals[key] += counts[key]

tickets["redacted_text"] = redacted_texts

print("\n========== PRIVACY AUDIT ==========")

for key, count in redaction_totals.items():
    print(f"{key}: {count}")

print(
    "Total substitutions:",
    sum(redaction_totals.values())
)

print("Remaining personal information: 0")

assert len(redacted_texts) == len(tickets)

# Verify every redacted ticket again.
for safe_text in redacted_texts:
    check_redaction(safe_text)


# --------------------------------------------------
# ROUTING RULES
# --------------------------------------------------

HUMAN_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
}


def add_routing(result):
    customer_id = str(result["customer_id"])

    customer = customer_lookup.get(
        customer_id,
        {"segment_name": "", "is_premium": False}
    )

    result["segment_name"] = customer["segment_name"]

    if result["status"] in HUMAN_STATUSES:
        result["route"] = "human_review"
        result["priority"] = "NORMAL"

    elif result["category"] == "fraud":
        result["route"] = "security_team"
        result["priority"] = "HIGH"

    elif customer["is_premium"]:
        result["route"] = "premium_desk"
        result["priority"] = "HIGH"

    else:
        result["route"] = "support_queue"
        result["priority"] = "NORMAL"

    return result


# --------------------------------------------------
# COMPLETE TRIAGE PIPELINE
# --------------------------------------------------

def run_pipeline(system_prompt, token_budget=None, run_name="run"):

    # Reset immediately before every independent full run.
    mock_llm.reset()

    running_tokens = 0
    results = []

    for index, ticket in tickets.iterrows():

        ticket_id = str(ticket["ticket_id"])
        customer_id = str(ticket["customer_id"])
        safe_text = ticket["redacted_text"]

        result = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": "",
            "category": "",
            "segment_name": "",
            "priority": "",
            "route": "",
            "tokens_used": 0,
            "true_category": str(ticket["true_category"]).strip().lower()
        }

        # Guardrail 1: empty ticket.
        if not safe_text.strip():
            result["status"] = "skipped_empty"
            results.append(add_routing(result))
            continue

        # Guardrail 2: long ticket.
        estimated = mock_llm.est_tokens(safe_text)

        if estimated > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(add_routing(result))
            continue

        # Budget guardrail: check before making a model call.
        if (
            token_budget is not None
            and running_tokens >= token_budget
        ):
            result["status"] = "budget_exhausted"
            results.append(add_routing(result))
            continue

        # Assert privacy immediately before the model call.
        check_redaction(safe_text)

        # User content is EXACTLY the redacted ticket text.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": safe_text}
        ]

        response = call_with_retry(messages, ticket_id)

        content = response.choices[0].message.content

        used_tokens = response.usage.total_tokens
        running_tokens += used_tokens

        category, status = validate_response(content)

        result["category"] = category
        result["status"] = status
        result["tokens_used"] = used_tokens

        results.append(add_routing(result))

    output = pd.DataFrame(results)

    print(f"\n========== {run_name.upper()} ==========")
    print("Tickets processed:", len(output))
    print("Total tokens:", running_tokens)

    print("\nStatus counts:")
    print(output["status"].value_counts().to_string())

    return output, running_tokens


# --------------------------------------------------
# METRICS AND CONFUSION TABLE
# --------------------------------------------------

def calculate_metrics(results, total_tokens, run_name):

    classified = results[
        results["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ].copy()

    if len(classified):
        correct = (
            classified["category"]
            == classified["true_category"]
        ).sum()

        accuracy = correct / len(classified)
    else:
        correct = 0
        accuracy = 0

    tokens_per_ticket = (
        total_tokens / len(classified)
        if len(classified)
        else 0
    )

    print(f"\n========== {run_name} METRICS ==========")
    print("Tickets classified:", len(classified))
    print(f"Correct classifications: {correct}")
    print(f"Accuracy: {accuracy:.4%}")
    print("Total tokens:", total_tokens)
    print(f"Tokens per classified ticket: {tokens_per_ticket:.2f}")

    return {
        "run": run_name,
        "classified": len(classified),
        "correct": int(correct),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_ticket": tokens_per_ticket
    }


# --------------------------------------------------
# RUN 1: ZERO-SHOT, NO BUDGET
# --------------------------------------------------

zero_results, zero_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=None,
    run_name="zero-shot comparison"
)

zero_metrics = calculate_metrics(
    zero_results, zero_tokens, "ZERO-SHOT"
)

zero_classified = zero_results[
    zero_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

print("\nZERO-SHOT CONFUSION TABLE")

confusion = pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"],
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False
).reindex(
    index=sorted(ALLOWED_CATEGORIES),
    columns=sorted(ALLOWED_CATEGORIES),
    fill_value=0
)

print(confusion)


# --------------------------------------------------
# RUN 2: FEW-SHOT, NO BUDGET
# --------------------------------------------------

few_results, few_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=None,
    run_name="few-shot comparison"
)

few_metrics = calculate_metrics(
    few_results, few_tokens, "FEW-SHOT"
)


# --------------------------------------------------
# COMPARE ZERO-SHOT AND FEW-SHOT
# --------------------------------------------------

print("\n========== PROMPT COMPARISON ==========")

print(
    pd.DataFrame([
        zero_metrics,
        few_metrics
    ]).to_string(index=False)
)

extra_tokens = few_tokens - zero_tokens

print("\nExtra few-shot tokens:", extra_tokens)
print(
    "Extra tokens per classified ticket:",
    round(
        extra_tokens / max(few_metrics["classified"], 1),
        2
    )
)

print(
    "Accuracy improvement in percentage points:",
    round(
        (few_metrics["accuracy"] - zero_metrics["accuracy"]) * 100,
        2
    )
)


# --------------------------------------------------
# RUN 3: ZERO-SHOT TOKEN BUDGET DEMONSTRATION
# --------------------------------------------------

DEMO_BUDGET = 6000

budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=DEMO_BUDGET,
    run_name="budget demonstration"
)

exhausted_count = (
    budget_results["status"] == "budget_exhausted"
).sum()

# If 6000 is not restrictive enough, repeat with 5000.
if exhausted_count == 0:

    DEMO_BUDGET = 5000

    print(
        "\nNo tickets exhausted at 6000 tokens. "
        "Repeating with a 5000-token budget."
    )

    budget_results, budget_tokens = run_pipeline(
        ZERO_SHOT_PROMPT,
        token_budget=DEMO_BUDGET,
        run_name="budget demonstration at 5000"
    )

    exhausted_count = (
        budget_results["status"] == "budget_exhausted"
    ).sum()

budget_results.drop(
    columns=["true_category"],
    errors="ignore"
).to_csv(
    OUTPUT_DIR / "triage_budget_demo.csv",
    index=False
)

print("\nBudget used:", DEMO_BUDGET)
print("Budget exhausted tickets:", exhausted_count)


# --------------------------------------------------
# RUN 4: FINAL FEW-SHOT TRIAGE
# --------------------------------------------------

final_results, final_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=100000,
    run_name="final few-shot triage"
)

final_results.drop(
    columns=["true_category"],
    errors="ignore"
)[[
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]].to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False
)


# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------

print("\n========== FINAL OUTPUT VALIDATION ==========")

print(
    "triage_results.csv rows:",
    len(pd.read_csv(OUTPUT_DIR / "triage_results.csv"))
)

print(
    "triage_budget_demo.csv rows:",
    len(pd.read_csv(OUTPUT_DIR / "triage_budget_demo.csv"))
)

assert len(final_results) == 164
assert len(budget_results) == 164
assert exhausted_count > 0

print("Both CSV files generated successfully.")


# --------------------------------------------------
# ADD PART C SUMMARY TO README
# --------------------------------------------------

readme_path = BASE_DIR / "README.md"

readme_section = f"""

## Part C — Support Ticket Triage with Guardrails

The support-ticket triage pipeline processes tickets in their original file order.

### Privacy and input guardrails

- Empty tickets are skipped.
- Tickets exceeding 400 estimated input tokens are flagged as long.
- Email addresses, phone numbers and order IDs are redacted before model calls.
- Redaction substitutions: {sum(redaction_totals.values())} total.
- Remaining matching personal information after redaction: 0.

### Prompt comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | {zero_metrics['classified']} | {few_metrics['classified']} |
| Correct classifications | {zero_metrics['correct']} | {few_metrics['correct']} |
| Accuracy | {zero_metrics['accuracy']:.2%} | {few_metrics['accuracy']:.2%} |
| Total tokens | {zero_metrics['total_tokens']} | {few_metrics['total_tokens']} |
| Tokens per classified ticket | {zero_metrics['tokens_per_ticket']:.2f} | {few_metrics['tokens_per_ticket']:.2f} |

The few-shot prompt used {extra_tokens} additional tokens compared with zero-shot.
Its accuracy changed from {zero_metrics['accuracy']:.2%} to
{few_metrics['accuracy']:.2%}. The additional prompt cost is worthwhile
when the accuracy improvement justifies the extra token usage, particularly
for sensitive categories such as fraud.

### Budget demonstration

The zero-shot budget demonstration used a token budget of {DEMO_BUDGET}.
It stopped before making further model calls for {exhausted_count} tickets.

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
"""

if readme_path.exists():
    existing = readme_path.read_text(encoding="utf-8")

    if "## Part C — Support Ticket Triage with Guardrails" not in existing:
        with readme_path.open("a", encoding="utf-8") as file:
            file.write(readme_section)
else:
    readme_path.write_text(
        "# Instamart Capstone\n" + readme_section,
        encoding="utf-8"
    )

print("README.md updated.")
print("\nPART C COMPLETED!")