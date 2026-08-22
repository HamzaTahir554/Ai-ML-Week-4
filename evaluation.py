import pandas as pd

from Chatbot import app


# ============================================================
# 1. LOAD GOLDEN QUESTIONS
# ============================================================

df = pd.read_csv("golden_questions.csv")


print("\nCSV columns found:")
print(df.columns.tolist())


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "question",
    "expected_answer"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nERROR: Missing column '{column}'.\n"
            f"Your CSV contains: {df.columns.tolist()}\n\n"
            f"Your CSV must contain:\n"
            f"question,expected_answer"
        )


# ============================================================
# 3. EVALUATION
# ============================================================

results = []

correct = 0


for index, row in df.iterrows():

    question = row["question"]

    expected = row["expected_answer"]


    print("\n" + "=" * 70)

    print(
        f"QUESTION {index + 1}/{len(df)}"
    )

    print("=" * 70)

    print(
        "\nQuestion:"
    )

    print(question)


    # ========================================================
    # RUN RAG AGENT
    # ========================================================

    result = app.invoke({

        "question": question,

        "context": "",

        "answer": ""

    })


    answer = result["answer"]


    print(
        "\nExpected Answer:"
    )

    print(expected)


    print(
        "\nGenerated Answer:"
    )

    print(answer)


    # ========================================================
    # MANUAL EVALUATION
    # ========================================================

    while True:

        score = input(
            "\nIs the answer correct? "
            "(1 = Yes, 0 = No): "
        )

        if score in ["0", "1"]:

            break

        print(
            "Please enter only 1 or 0."
        )


    score = int(score)


    if score == 1:

        correct += 1


    results.append({

        "question": question,

        "expected_answer": expected,

        "generated_answer": answer,

        "correct": score

    })


# ============================================================
# 4. CALCULATE ACCURACY
# ============================================================

total_questions = len(df)


accuracy = (
    correct / total_questions
) * 100


print("\n\n")
print("=" * 70)

print("FINAL EVALUATION")

print("=" * 70)

print(
    f"Total Questions : {total_questions}"
)

print(
    f"Correct Answers : {correct}"
)

print(
    f"Wrong Answers   : {total_questions - correct}"
)

print(
    f"Accuracy        : {accuracy:.2f}%"
)


# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    "evaluation_results.csv",
    index=False
)


print(
    "\nEvaluation results saved to:"
)

print(
    "evaluation_results.csv"
)