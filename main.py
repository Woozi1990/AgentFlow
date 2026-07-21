import uuid

from agents import Agent
from models import OpenAIModel

if __name__ == "__main__":
    llm = OpenAIModel()
    agent = Agent(llm)
    output_path = "output.mp3"
    image_path = "data/example.png"
    thread_id = str(uuid.uuid4())
    while True:
        user_input = input("\nUser:").strip()

        if user_input.lower() == "exit":
            break

        agent.run(user_input, image_path, output_path, thread_id)
        print()

    # print("\n========== Final Answer ==========")
    # print(final_answer)
