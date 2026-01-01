#!/usr/bin/env python3
"""
Exam Correction Agent for CSCI 101 Lab Exam
Uses DigitalOcean Inference API (llama3.3-70b-instruct) for soft grading
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

EXCEL_FILE = "CSCI 101 Lab Exam Submission Form (Responses).xlsx"
OUTPUT_FILE = "exam_results.xlsx"
SECTIONS_TO_GRADE = ["WNL8", "WNL10"]

# Point distribution
POINTS = {
    "WNL8": {
        "Q1": {"total": 3, "parts": {"full": 3}},  # Q1 is graded as whole
        "Q2": {"total": 3, "parts": {"a": 2, "b": 1}},
    },
    "WNL10": {
        "Q1": {"total": 3, "parts": {"a": 2, "b": 1}},
        "Q2": {"total": 3, "parts": {"full": 3}},  # Q2 is graded as whole
    },
}

# ============================================================================
# EXAM QUESTIONS
# ============================================================================

EXAM_QUESTIONS = {
    "WNL8": {
        "Q1": """
Question 1 (3 points) - Count_digits Function:

Part a) Write a function called Count_digits(string) that takes a string as input and
returns the number of digits in the string.
- The function must NOT read input from the user
- The function must NOT print anything
- The function must return an integer (the count of digits)

Part b) Write a script that:
- Asks the user to enter a sentence
- Calls Count_digits() with the user's input
- Prints the number of digits found in the sentence
""",
        "Q2": """
Question 2 (3 points) - Score Dictionary:

Write a Python program that:
1. Prompts the user to enter a comma-separated list of participant names
2. Prompts the user to enter a comma-separated list of their scores
3. Builds a dictionary with names as keys and scores as values
4. Prints the resulting dictionary
5. Determines and prints the top scorer with their score
""",
    },
    "WNL10": {
        "Q1": """
Question 1 (3 points) - drawTriangle Function:

Part a) Write a function called drawTriangle(n) that takes an integer n and prints
a triangle pattern of stars. For example, if n=5:
*****
****
***
**
*

The function only prints the pattern (no return value).
Assume n is always a positive integer.

Part b) Write a script that:
- Asks the user to enter a number
- Calls drawTriangle() with the user's input
- Displays the triangle pattern
""",
        "Q2": """
Question 2 (3 points) - commonElements Function:

Part a) Write a function called commonElements(list1, list2) that:
- Takes two lists as input
- Returns a list of distinct elements that appear in BOTH lists
- Returns an empty list if no common elements exist

Part b) Write a script that:
- Prompts user to enter two lists (space-separated values)
- Calls commonElements() with both lists
- Prints the common elements or an appropriate message if empty
""",
    },
}

# ============================================================================
# GRADING PROMPT TEMPLATE
# ============================================================================

GRADING_PROMPT_TEMPLATE = """
You are a gentle and encouraging programming instructor grading a CSCI 101 introductory Python exam.

## GRADING PHILOSOPHY - SOFT GRADING
- Be LENIENT and give partial credit generously
- Award points for ANY correct concepts, even if implementation is incomplete
- Give credit for correct logic even with syntax errors
- Recognize effort and understanding of the problem
- If the student shows they understood the question, give partial credit
- Only give zero if the answer is completely unrelated or empty

## EXAM SECTION: {section}

## POINT DISTRIBUTION:
{point_info}

================================================================================
## QUESTION 1:
================================================================================
{q1_text}

### STUDENT'S ANSWER TO QUESTION 1 (Part A - the function):
```python
{q1_a}
```

### STUDENT'S ANSWER TO QUESTION 1 (Part B - the script):
```python
{q1_b}
```

================================================================================
## QUESTION 2:
================================================================================
{q2_text}

### STUDENT'S ANSWER TO QUESTION 2:
```python
{q2_a}
```

================================================================================
## YOUR TASK:
================================================================================
IMPORTANT: Grade each question based ONLY on the student's answer shown directly below that question.
- For Q1: Look at the code under "STUDENT'S ANSWER TO QUESTION 1" and grade it against Question 1 requirements
- For Q2: Look at the code under "STUDENT'S ANSWER TO QUESTION 2" and grade it against Question 2 requirements

DO NOT mix up the questions. Each answer must be graded against its own question.

## RESPONSE FORMAT (respond ONLY with valid JSON, no other text):
{{
    "Q1": {{
        "feedback": "Feedback about the student's Question 1 answer",
        "correct_parts": ["list", "of", "correct", "concepts"],
        "points_earned": <number between 0 and {q1_max}>,
        "max_points": {q1_max}
    }},
    "Q2": {{
        "feedback": "Feedback about the student's Question 2 answer",
        "correct_parts": ["list", "of", "correct", "concepts"],
        "points_earned": <number between 0 and {q2_max}>,
        "max_points": {q2_max}
    }},
    "total_points": <sum of points_earned>,
    "max_total": {total_max},
    "overall_comment": "One encouraging sentence about their work"
}}
"""


# ============================================================================
# DIGITALOCEAN INFERENCE API
# ============================================================================

API_URL = "https://inference.do-ai.run/v1/chat/completions"
MODEL_NAME = "llama3.3-70b-instruct"


def get_api_key():
    """Get API key from environment"""
    api_key = os.environ.get("MODEL_ACCESS_KEY")
    if not api_key:
        raise ValueError("MODEL_ACCESS_KEY environment variable not set")
    return api_key


def call_inference_api(api_key, messages, max_tokens=1500):
    """Call DigitalOcean Inference API directly via HTTP"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,  # Lower temperature for consistent grading
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()


def grade_student(api_key, section, q1_a, q1_b, q2_a):
    """Grade a single student's exam using the AI"""

    questions = EXAM_QUESTIONS[section]
    points = POINTS[section]

    q1_max = points["Q1"]["total"]
    q2_max = points["Q2"]["total"]
    total_max = q1_max + q2_max

    # Build point info string
    if section == "WNL8":
        point_info = "Q1: 3 points (graded as whole)\nQ2: 3 points (part a: 2 points, part b: 1 point)"
    else:  # WNL10
        point_info = "Q1: 3 points (part a: 2 points, part b: 1 point)\nQ2: 3 points (graded as whole)"

    prompt = GRADING_PROMPT_TEMPLATE.format(
        section=section,
        q1_text=questions['Q1'],
        q2_text=questions['Q2'],
        point_info=point_info,
        q1_a=q1_a or "(No answer provided)",
        q1_b=q1_b or "(No answer provided)",
        q2_a=q2_a or "(No answer provided)",
        q1_max=q1_max,
        q2_max=q2_max,
        total_max=total_max,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful programming instructor. Always respond with valid JSON only, no markdown formatting."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response_data = call_inference_api(api_key, messages)
        result_text = response_data["choices"][0]["message"]["content"].strip()

        # Try to parse JSON from response
        # Handle case where response might have markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result_text = result_text.strip()
        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        print(f"  Warning: Could not parse AI response as JSON: {e}")
        print(f"  Raw response: {result_text[:500]}")
        # Return a default grading if parsing fails
        return {
            "Q1": {"feedback": "Grading error - manual review needed", "correct_parts": [], "points_earned": 0, "max_points": q1_max},
            "Q2": {"feedback": "Grading error - manual review needed", "correct_parts": [], "points_earned": 0, "max_points": q2_max},
            "total_points": 0,
            "max_total": total_max,
            "overall_comment": "This submission needs manual review due to a grading error."
        }
    except Exception as e:
        print(f"  Error calling API: {e}")
        raise


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def load_students(filepath):
    """Load and filter students from Excel file"""
    print(f"Loading students from: {filepath}")
    df = pd.read_excel(filepath)

    # Rename columns for easier access
    df.columns = [
        "Timestamp", "Email_Address", "Name", "ID", "Email", "Section_Code",
        "Q1_a", "Q1_a_file", "Q1_b", "Q1_b_file", "Q2_a", "Q2_b_file"
    ]

    # Filter for our sections
    df_filtered = df[df["Section_Code"].isin(SECTIONS_TO_GRADE)].copy()

    print(f"Found {len(df_filtered)} students in sections {SECTIONS_TO_GRADE}")

    # Show breakdown
    for section in SECTIONS_TO_GRADE:
        count = len(df_filtered[df_filtered["Section_Code"] == section])
        print(f"  - {section}: {count} students")

    return df_filtered


def process_all_students(df):
    """Process all students and return results"""
    api_key = get_api_key()
    results = []

    total = len(df)
    print(f"\nStarting grading for {total} students...")
    print(f"Using model: {MODEL_NAME}")
    print("=" * 60)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        name = row["Name"]
        student_id = row["ID"]
        section = row["Section_Code"]

        print(f"\n[{idx}/{total}] Grading: {name} (Section: {section})")

        # Get student answers (handle NaN)
        q1_a = row["Q1_a"] if pd.notna(row["Q1_a"]) else ""
        q1_b = row["Q1_b"] if pd.notna(row["Q1_b"]) else ""
        q2_a = row["Q2_a"] if pd.notna(row["Q2_a"]) else ""

        # Grade the student
        try:
            grading = grade_student(api_key, section, q1_a, q1_b, q2_a)

            result = {
                "Name": name,
                "ID": student_id,
                "Email": row["Email_Address"],
                "Section": section,
                "Q1_Code_a": q1_a[:500] if q1_a else "",  # Truncate for output
                "Q1_Code_b": q1_b[:500] if q1_b else "",
                "Q2_Code": q2_a[:500] if q2_a else "",
                "Q1_Feedback": grading["Q1"]["feedback"],
                "Q1_Correct_Parts": ", ".join(grading["Q1"]["correct_parts"]),
                "Q1_Points": grading["Q1"]["points_earned"],
                "Q1_Max": grading["Q1"]["max_points"],
                "Q2_Feedback": grading["Q2"]["feedback"],
                "Q2_Correct_Parts": ", ".join(grading["Q2"]["correct_parts"]),
                "Q2_Points": grading["Q2"]["points_earned"],
                "Q2_Max": grading["Q2"]["max_points"],
                "Total_Points": grading["total_points"],
                "Max_Points": grading["max_total"],
                "Percentage": round(grading["total_points"] / grading["max_total"] * 100, 1),
                "Overall_Comment": grading["overall_comment"],
            }

            print(f"  -> Score: {grading['total_points']}/{grading['max_total']} ({result['Percentage']}%)")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            result = {
                "Name": name,
                "ID": student_id,
                "Email": row["Email_Address"],
                "Section": section,
                "Q1_Code_a": q1_a[:500] if q1_a else "",
                "Q1_Code_b": q1_b[:500] if q1_b else "",
                "Q2_Code": q2_a[:500] if q2_a else "",
                "Q1_Feedback": f"ERROR: {e}",
                "Q1_Correct_Parts": "",
                "Q1_Points": 0,
                "Q1_Max": POINTS[section]["Q1"]["total"],
                "Q2_Feedback": f"ERROR: {e}",
                "Q2_Correct_Parts": "",
                "Q2_Points": 0,
                "Q2_Max": POINTS[section]["Q2"]["total"],
                "Total_Points": 0,
                "Max_Points": 6,
                "Percentage": 0,
                "Overall_Comment": "Error during grading - needs manual review",
            }

        results.append(result)

    return results


def save_results(results, output_path):
    """Save results to Excel with multiple sheets"""
    print(f"\nSaving results to: {output_path}")

    df_results = pd.DataFrame(results)

    # Create Excel writer with multiple sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Full results sheet
        df_results.to_excel(writer, sheet_name="Full Results", index=False)

        # Summary sheet (just grades)
        summary_cols = ["Name", "ID", "Section", "Q1_Points", "Q2_Points",
                       "Total_Points", "Max_Points", "Percentage"]
        df_summary = df_results[summary_cols].copy()
        df_summary.to_excel(writer, sheet_name="Grade Summary", index=False)

        # Per-section sheets
        for section in SECTIONS_TO_GRADE:
            df_section = df_results[df_results["Section"] == section]
            if not df_section.empty:
                df_section.to_excel(writer, sheet_name=f"{section} Results", index=False)

    print(f"Saved {len(results)} student results")

    # Print summary stats
    print("\n" + "=" * 60)
    print("GRADING SUMMARY")
    print("=" * 60)

    for section in SECTIONS_TO_GRADE:
        section_results = [r for r in results if r["Section"] == section]
        if section_results:
            avg = sum(r["Percentage"] for r in section_results) / len(section_results)
            print(f"\n{section}:")
            print(f"  Students: {len(section_results)}")
            print(f"  Average: {avg:.1f}%")
            print(f"  Highest: {max(r['Percentage'] for r in section_results):.1f}%")
            print(f"  Lowest: {min(r['Percentage'] for r in section_results):.1f}%")


def main():
    """Main entry point"""
    print("=" * 60)
    print("CSCI 101 Exam Correction Agent")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, EXCEL_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    # Load students
    df = load_students(excel_path)

    if df.empty:
        print("No students found to grade!")
        return

    # Process all students
    results = process_all_students(df)

    # Save results
    save_results(results, output_path)

    print("\n" + "=" * 60)
    print("GRADING COMPLETE!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
