import inspect
import mock_llm

print("========== AGENT FUNCTION ==========")
print(inspect.getsource(mock_llm._agent))

print("\n========== POLICY FUNCTION ==========")
print(inspect.getsource(mock_llm._policy))

print("\n========== MOCK LLM FILE ==========")
print(inspect.getsource(mock_llm))