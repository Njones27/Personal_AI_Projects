from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, set_tracing_disabled, AsyncOpenAI, set_default_openai_client, set_default_openai_api
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
import subprocess
import time
import urllib.request
import urllib.error
import socket

load_dotenv(find_dotenv(), override=False)

set_tracing_disabled(True)

# Ensure Ollama is running locally. If not, start it in the background.
def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ollama_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/api/tags", timeout=1) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def start_ollama_if_needed(host: str = "127.0.0.1", port: int = 11434, wait_seconds: int = 1) -> None:
    base = f"http://{host}:{port}"
    # Fast path: already running?
    if _is_port_open(host, port) and _ollama_ready(base):
        return

    # Not running: try to start in the background.
    # Note: `ollama serve` runs in the foreground; we detach it so this script can continue.
    # stdout/stderr are suppressed; remove `DEVNULL` if you want logs.
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Could not start 'ollama serve': the 'ollama' binary was not found in PATH. "
            "Install Ollama or add it to your PATH."
        )

    # Wait for the server to become ready
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _is_port_open(host, port) and _ollama_ready(base):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(
            f"Started 'ollama serve' but it did not become ready on {base} within {wait_seconds}s."
        )

# Start Ollama if needed before configuring the OpenAI-compatible client
start_ollama_if_needed()

# Route the SDK to a self-hosted OpenAI-compatible server (e.g., Ollama/LM Studio/vLLM)
set_default_openai_client(
    AsyncOpenAI(
        base_url="http://localhost:11434/v1",  # change if your server runs elsewhere
        api_key="local-anything"               # many local servers accept any token
    )
)
# Most local servers implement Chat Completions, not Responses
set_default_openai_api("chat_completions")

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
    model="gpt-oss:20b",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    model="gpt-oss:20b",
)


history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
    model="gpt-oss:20b",
)

local_tutor_agent = Agent(
    name="Local Tutor",
    handoff_description="Local OSS model via LiteLLM/Ollama",
    instructions="Be helpful and show steps.",
    model="gpt-oss:20b",
)


async def homework_guardrail(ctx, agent, input_data):
    # Simple local guardrail: detect if the user is asking about homework without calling an LLM
    text = input_data if isinstance(input_data, str) else str(input_data)
    lower = text.lower()
    keywords = [
        "homework", "assignment", "problem set", "pset", "quiz", "exam",
        "worksheet", "take-home", "due", "question 1", "q1"
    ]
    is_hw = any(k in lower for k in keywords)
    reasoning = (
        "keyword match: " + ", ".join([k for k in keywords if k in lower])
        if is_hw else "no homework-related keywords detected"
    )
    final_output = HomeworkOutput(is_homework=is_hw, reasoning=reasoning)
    return GuardrailFunctionOutput(output_info=final_output, tripwire_triggered=final_output.is_homework)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent, local_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
    model="gpt-oss:20b",
)

async def main():
    # if not os.getenv("OPENAI_API_KEY"):       # locally hosting (free)
    #     pass
    # Example 1: History question
    try:
        result = await Runner.run(triage_agent, "who was the first president of the united states?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

    # Example 2: General/philosophical question
    try:
        result = await Runner.run(triage_agent, "What is the meaning of life?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

if __name__ == "__main__":
    asyncio.run(main())