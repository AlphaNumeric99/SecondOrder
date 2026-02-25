"""Quick test to verify LLM calls are working."""
import asyncio

from app.llm_client import client, get_model


async def test_llm_call():
    """Test a simple LLM call to verify connectivity."""
    model = get_model()
    print(f"Using model: {model}")

    try:
        response = await client().messages.create(
            model=model,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Say 'Hello' if you can hear me."}],
            max_tokens=50,
        )
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_llm_call())
    print(f"LLM call {'succeeded' if result else 'failed'}")
