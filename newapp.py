import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import random
import re
import dbauthentication  # Import the database authentication module

# Initialize OpenAI LLM
llm = OpenAI(
    openai_api_key="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",  # Replace with your actual OpenAI API key
    model_name="gpt-3.5-turbo-instruct",
    temperature=1.0
)

# Ensure tables exist
dbauthentication.create_tables()

if "current_page" not in st.session_state:
    st.session_state.current_page = "Login"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ------------------------------
# Sidebar Navigation with Buttons
# ------------------------------
st.sidebar.title("📌 Navigation")

if st.sidebar.button("🔑 Login", key="login_page"):
    st.session_state.current_page = "Login"
if st.sidebar.button("📝 Register", key="register_page"):
    st.session_state.current_page = "Register"

if st.session_state.logged_in:
    if st.sidebar.button("🔍 Mind Map & Summary", key="summary_page"):
        st.session_state.current_page = "Mind Map & Summary"
    if st.sidebar.button("🧩 Quiz", key="quiz_page"):
        st.session_state.current_page = "Quiz"
    if st.sidebar.button("👤 Profile", key="profile_page"):
        st.session_state.current_page = "Profile"
    if st.sidebar.button("🚪 Logout", key="logout_page"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.current_page = "Login"

# ------------------------------
# Functions for Processing
# ------------------------------
def generate_summary(text):
    prompt_template = PromptTemplate(
        input_variables=["lecture_notes"],
        template=(
            "Provide a concise summary of the following text:\n\n"
            "{lecture_notes}\n\n"
            "Keep it short, clear, and informative."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(text)

def extract_main_topic(text):
    prompt_template = PromptTemplate(
        input_variables=["lecture_notes"],
        template=(
            "Identify the main topic of the text in one word:\n\n"
            "Text: {lecture_notes}\n"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(text).strip()
    return result.split()[0] if result else ""

def generate_quiz_questions(topic):
    prompt_template = PromptTemplate(
        input_variables=["topic"],
        template=(
            "Generate five multiple-choice quiz questions on the topic '{topic}'. "
            "Each question MUST have exactly four answer choices, labeled a), b), c), and d). "
            "Ensure that the format is STRICTLY as follows:\n\n"
            "1. Question text\n"
            "   a) Option 1\n"
            "   b) Option 2\n"
            "   c) Option 3\n"
            "   d) Option 4\n"
            "   Answer: correct option\n"
            "Repeat for 5 questions."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(topic)

def parse_quiz(quiz_text):
    """Parses quiz text and ensures each question has 4 choices."""
    questions = []
    lines = quiz_text.strip().split("\n")
    question = None

    question_pattern = re.compile(r"^\d+\.\s*(.*)")
    option_pattern = re.compile(r"^[a-d]\)\s*(.*)")
    answer_pattern = re.compile(r"^Answer:\s*(.*)")

    for line in lines:
        line = line.strip()
        if question_pattern.match(line):
            if question and len(question["options"]) == 4:
                questions.append(question)
            question = {"question": question_pattern.match(line).group(1), "options": [], "answer": ""}
        elif option_pattern.match(line) and question:
            question["options"].append(line)
        elif answer_pattern.match(line) and question:
            question["answer"] = answer_pattern.match(line).group(1)

    if question and len(question["options"]) == 4:
        questions.append(question)

    return questions

def plot_progress(quiz_history):
    """
    Create a combined bar and line chart showing quiz progress.
    
    Args:
        quiz_history: list of tuples (id, username, quiz_date, total_questions, score, percentage)
    Returns:
        A matplotlib figure.
    """
    attempts = list(range(1, len(quiz_history) + 1))
    percentages = [record[5] for record in quiz_history]  # record[5] is percentage
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(attempts, percentages, alpha=0.6, label="Score Percentage")
    ax.plot(attempts, percentages, marker='o', color='red', label="Trend")
    ax.set_xlabel("Quiz Attempt")
    ax.set_ylabel("Score Percentage")
    ax.set_title("Quiz Progress Over Time")
    ax.set_xticks(attempts)
    ax.set_ylim(0, 100)
    ax.legend()
    return fig

# ------------------------------
# Login Page
# ------------------------------
if st.session_state.current_page == "Login":
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if dbauthentication.verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.current_page = "Mind Map & Summary"
            st.success("✅ Login successful! Navigate using the sidebar.")
        else:
            st.error("❌ Invalid username or password.")

# ------------------------------
# Registration Page
# ------------------------------
elif st.session_state.current_page == "Register":
    st.title("📝 Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if dbauthentication.add_user(username, email, password):
            st.success("✅ Registration successful! Please log in.")
            st.session_state.current_page = "Login"
        else:
            st.error("❌ Username or email already exists.")

# ------------------------------
# Mind Map & Summary Page
# ------------------------------
elif st.session_state.logged_in and st.session_state.current_page == "Mind Map & Summary":
    st.title("🧠 Mind Map & Summary")
    user_input = st.text_area("📜 Paste the transcribed text here:", height=200)

    if st.button("🚀 Generate Mind Map and Quiz"):
        if user_input.strip():
            with st.spinner("⚡ Processing..."):
                summary = generate_summary(user_input)
                central_topic = extract_main_topic(user_input)
                quiz_text = generate_quiz_questions(central_topic)
                questions = parse_quiz(quiz_text)

                if len(questions) != 5 or any(len(q["options"]) != 4 for q in questions):
                    st.error("⚠️ AI-generated quiz did not meet the expected format. Please try again.")
                else:
                    st.session_state.summary = summary
                    st.session_state.central_topic = central_topic
                    st.session_state.questions = questions
                    st.success("✅ Data generated! Click the **Quiz** button in the sidebar.")

    if "summary" in st.session_state:
        st.subheader("📄 Summary")
        st.write(st.session_state.summary)
        st.subheader("📝 Main Topic")
        st.write(f"**{st.session_state.central_topic}**")

# ------------------------------
# Quiz Page
# ------------------------------
elif st.session_state.logged_in and st.session_state.current_page == "Quiz":
    st.title("🧩 Quiz")

    # Regenerate Quiz Button
    if st.button("🔄 Regenerate Quiz"):
        if "central_topic" in st.session_state:
            quiz_text = generate_quiz_questions(st.session_state.central_topic)
            new_questions = parse_quiz(quiz_text)
            if len(new_questions) != 5 or any(len(q["options"]) != 4 for q in new_questions):
                st.error("⚠️ AI-generated quiz did not meet the expected format. Please try again.")
            else:
                st.session_state.questions = new_questions
                st.success("✅ New quiz generated!")
        else:
            st.error("Please generate the Mind Map & Summary first.")

    if "questions" in st.session_state:
        questions = st.session_state.questions
        with st.form(key="quiz_form"):
            quiz_answers = {}
            for idx, q in enumerate(questions):
                st.write(f"**{idx + 1}. {q['question']}**")
                quiz_answers[idx] = st.radio(f"Select an answer for Question {idx + 1}:", q["options"], key=f"q{idx}")
            submit_quiz = st.form_submit_button("🎯 Submit Quiz")

        if submit_quiz:
            score = 0
            for idx, q in enumerate(questions):
                correct = q["answer"]
                selected = quiz_answers.get(idx)
                if selected and correct and selected.startswith(correct[0]):
                    score += 1

            percentage = (score / len(questions)) * 100
            # Update aggregated stats and log this quiz attempt.
            dbauthentication.update_quiz_stats(
                st.session_state.username, 1, len(questions), score, percentage
            )
            st.success(f"🏆 **Your Score: {score}/{len(questions)} ({percentage:.2f}%)**")

    else:
        st.info("🔍 Please generate the **Mind Map & Quiz** first from the sidebar.")

# ------------------------------
# Profile Page
# ------------------------------
elif st.session_state.logged_in and st.session_state.current_page == "Profile":
    st.title("👤 Profile")
    st.write(f"**Username:** {st.session_state.username}")
    
    # Retrieve and display the user's aggregated quiz statistics.
    stats = dbauthentication.get_user_stats(st.session_state.username)
    
    if stats:
        st.subheader("📊 Quiz Statistics")
        st.write(f"**Total Quizzes Taken:** {stats.get('total_quizzes', 0)}")
        st.write(f"**Total Questions Answered:** {stats.get('total_questions', 0)}")
        st.write(f"**Total Correct Answers:** {stats.get('total_correct', 0)}")
        st.write(f"**Average Score (%):** {stats.get('average_percentage', 0):.2f}%")
    else:
        st.info("No quiz statistics available yet.")
    
    # Retrieve quiz history and plot the progress graph if records exist.
    quiz_history = dbauthentication.get_quiz_history(st.session_state.username)
    if quiz_history:
        st.subheader("📈 Quiz Progress")
        fig = plot_progress(quiz_history)
        st.pyplot(fig)
    else:
        st.info("No quiz attempts recorded yet.")