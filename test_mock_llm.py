import mock_llm

messages = [
    {
        "role": "system",
        "content": "CLASSIFY customer support tickets"
    },
    {
        "role": "user",
        "content": "I was charged twice for my order. Please refund me."
    }
]

response = mock_llm.chat(messages)

print("CUSTOMER CLASSIFICATION:")
print(response.choices[0].message.content)