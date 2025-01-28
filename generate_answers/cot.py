from llm_config import *

system_prompt_before_context = '''When necessary, you should decompose the USER QUESTION to FOLLOW UP QUESTIONs and answer the FOLLOW UP QUESTIONs according to the context information below but not your prior knowledge'''

self_ask_prompt = '''Now I'll provide some EXAMPLE QUESTIONs below, you should follow the anwering pattern
EXAMPLE QUESTION: Who lived longer, Muhammad Ali or Alan Turing?
Are FOLLOW UP QUESTIONs needed here: Yes.
FOLLOW UP QUESTION: How old was Muhammad Ali when he died?
INTERMEDIATE ANSWER: Muhammad Ali was 74 years old when he died.
FOLLOW UP QUESTION: How old was Alan Turing when he died?
INTERMEDIATE ANSWER: Alan Turing was 41 years old when he died.
FINAL ANSWER: Muhammad Ali

EXAMPLE QUESTION: What was the scoring total for boston celtics on 2023-05-29?
Are FOLLOW UP QUESTIONs needed here: No.
FINAL ANSWER: 84

EXAMPLE QUESTION: When was the founder of craigslist born?
Are FOLLOW UP QUESTIONs needed here: Yes.
FOLLOW UP QUESTION: Who was the founder of craigslist?
INTERMEDIATE ANSWER: Craigslist was founded by Craig Newmark.
FOLLOW UP QUESTION: When was Craig Newmark born?
INTERMEDIATE ANSWER: Craig Newmark was born on December 6, 1952.
FINAL ANSWER: December 6, 1952

EXAMPLE QUESTION: Who was the maternal grandfather of George Washington?
Are FOLLOW UP QUESTIONs needed here: Yes.
FOLLOW UP QUESTION: Who was the mother of George Washington?
INTERMEDIATE ANSWER: The mother of George Washington was Mary Ball Washington.
FOLLOW UP QUESTION: Who was the father of Mary Ball Washington?
INTERMEDIATE ANSWER: The father of Mary Ball Washington was Joseph Ball.
FINAL ANSWER: Joseph Ball

EXAMPLE QUESTION: Are both the directors of Jaws and Casino Royale from the same country?
Are FOLLOW UP QUESTIONs needed here: Yes.
FOLLOW UP QUESTION: Who is the director of Jaws?
INTERMEDIATE ANSWER: The director of Jaws is Steven Spielberg.
FOLLOW UP QUESTION: Where is Steven Spielberg from?
INTERMEDIATE ANSWER: The United States.
FOLLOW UP QUESTION: Who is the director of Casino Royale?
INTERMEDIATE ANSWER: The director of Casino Royale is Martin Campbell.
FOLLOW UP QUESTION: Where is Martin Campbell from?
INTERMEDIATE ANSWER: New Zealand.
FINAL ANSWER: No

EXAMPLE QUESTION: Who among patricia place and lana clarkson is younger?
Are FOLLOW UP QUESTIONs needed here: No.
FINAL ANSWER: Lana Clarkson

USER QUESTION: '''


def llm_adapter(contexts, query):
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt_before_context + "\n" + str(contexts)},
            {"role": "user", "content": self_ask_prompt + query}
        ]
    )

    answer = response.choices[0].message.content
    print(f"Full answer from self_ask_context.llm_adapter: {answer}")

    clean_answer = extract_answer(answer)

    return clean_answer


def extract_answer(generated):
    if generated is None:
        return generated
    
    if "FINAL ANSWER:" not in generated:
        return generated

    return generated.split("FINAL ANSWER:")[-1].strip()