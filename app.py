import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Toxicity Classifier",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Toxicity Classifier")
st.markdown("### Welcome to the Lyzr Toxicity Classifier!")

query=st.text_input("Enter your comment: ")

open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

def toxicity_classifier(query):

    toxicity_agent = Agent(
            role="toxicity expert",
            prompt_persona=f"You are an Expert toxicity finder. Your task is to find whether the {query} is toxic or not"
        )

    prompt=f"""you are a toxicity classifier and you have to find whether {query} is toxic or not.[!IMPORTANT]ONLY ANSWER WHETHER SENTENCE IS TOXIC OR NOT.nothing else apart from this"""

    toxicity_task  =  Task(
        name="Toxicity Classifier",
        model=open_ai_text_completion_model,
        agent=toxicity_agent,
        instructions=prompt,
    )

    output = LinearSyncPipeline(
        name="Toxicity Pipline",
        completion_message="pipeline completed",
        tasks=[
              toxicity_task
        ],
    ).run()

    answer = output[0]['task_output']

    return answer

if st.button("Solve"):
    solution = toxicity_classifier(query)
    st.markdown(solution)

