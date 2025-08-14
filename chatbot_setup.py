import os
import json
import json5
import unicodedata
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
import concurrent.futures
import time

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

client = openai.OpenAI()

# Custom exception for quota exceeded
class QuotaExceededException(Exception):
    def __init__(self, message="OpenAI API quota exceeded"):
        self.message = message
        super().__init__(self.message)

# Function to check if the API key is valid
def check_api_key_validity() -> bool:
    try:
        # Test the API key by making a simple API call
        response = client.models.list()
        if response:
            print("API key is valid. Proceeding with the task...")
            return True
    except openai.AuthenticationError:
        print("Error: The API key is invalid. Please check your API key and try again.")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

# Updated key maps to maintain keys as specified in English
KEY_MAPS = {
    "English": {
        "question": "Question", 
        "discipline": "Discipline", 
        "language": "Language", 
        "grade": "Grade", 
        "explanation": "Explanation", 
        "competition_name": "Competition Name", 
        "competition_year": "Competition Year", 
        "question_number": "Question Number", 
        "difficulty": "Difficulty", 
        "question_type": "Question Type", 
        "answer": "Answer"
    }
}

# Extra possible variants for keys across languages
FALLBACK_KEYS = {
    "question": ["Question", "question"],
    "answer": ["Answer", "answer"],
    "question_type": ["Question Type", "question type"],
    "language": ["Language", "language"],
    "discipline": ["Discipline", "discipline"],
    "grade": ["Grade", "grade"],
    "explanation": ["Explanation", "explanation"],
    "competition_name": ["Competition Name", "competition name"],
    "competition_year": ["Competition Year", "competition year"],
    "question_number": ["Question Number", "question number"],
    "difficulty": ["Difficulty", "difficulty"]
}

# Normalize keys to handle different cases or accents
def normalize_key(key: str) -> str:
    return unicodedata.normalize("NFKC", key.strip().lower())

# Get the key map for a language
def get_key_map_for_language(language: str) -> Dict[str, str]:
    return KEY_MAPS.get("English", KEY_MAPS["English"])

# Map the record fields based on key maps
def map_record_fields(record: Dict[str, Any], key_map: Dict[str, str]) -> Dict[str, Any]:
    normalized_record = {normalize_key(k): v for k, v in record.items()}
    mapped = {}
    for eng_key, local_key in key_map.items():
        val = normalized_record.get(normalize_key(local_key), "")
        if not val:
            for alt_key in FALLBACK_KEYS.get(eng_key, []):
                if normalize_key(alt_key) in normalized_record:
                    val = normalized_record[normalize_key(alt_key)]
                    break
        mapped[eng_key] = val
    return mapped

# Construct system prompt for OpenAI API based on question type
def get_system_prompt(question_type: str) -> str:
    base = (
        "You are an expert educator, grader, and language expert specializing in mathematics, physics, and theoretical subjects. "
        "Your task is to evaluate the correctness of a given question and its provided answer in any language, including but not limited to English, Spanish, French, Chinese, Arabic, Russian, Hindi, and others. "
        "Follow these steps precisely, regardless of the language:\n"
        "1. Assess the provided answer for accuracy based on the given question. Ensure to check for logical consistency, correctness in mathematical or scientific reasoning, and clarity in the language used.\n"
        "2. If the answer is incorrect or lacks completeness, provide a detailed correction. Ensure that the **Explanation** contains the step-by-step process to solve the problem, while the **Answer** is the final, correct solution. Make sure to preserve the technical rigor and conceptual clarity while correcting, but write it in a natural, conversational tone as if you were explaining it to a student.\n"
        "3. Identify any ambiguous language or missing information in the answer. If the question involves mathematical or scientific formulas, preserve and correct them with exact LaTeX formatting.\n"
        "4. If needed, clarify the wording of the answer to ensure it is precise and unambiguous, without altering the meaning or changing the core content.\n"
        "5. Consider linguistic nuances in different languages and ensure that translations or explanations respect the integrity of the original content. Adapt your reasoning style to the conventions and standards of each language used in the question and answer.\n"
        "6. If the answer includes any mathematical or scientific symbols, LaTeX expressions, or formulas, preserve them exactly and correct only when necessary.\n"
        "Return valid JSON ONLY, in the following format:\n"
        "{\"valid\": bool, \"reason\": str, \"corrected_answer\": str or null, \"corrected_explanation\": str or null}\n"
        "- \"valid\": True if the answer is correct, False if it's not.\n"
        "- \"reason\": Provide a clear explanation of why the answer is correct or incorrect. Be as specific as possible, pointing out any errors in logic, calculation, or phrasing.\n"
        "- \"corrected_answer\": Provide the corrected final answer if the original answer was wrong. If the answer was correct, this should be null.\n"
        "- \"corrected_explanation\": Provide the corrected explanation if the original explanation was wrong. If the explanation was correct, this should be null.\n"
        "Ensure that your reasoning is clear, concise, and well-structured. If corrections are needed, provide a complete and accurate solution. Always preserve the original intent and meaning of the answer while making corrections, but write in a natural, human-like tone as if you're talking to a student."
    )

    qt = question_type.lower()
    if qt in ("explanation", "exp"): return base + " The answer should be a detailed explanation in a conversational and clear style."
    elif qt in ("short answer", "short"): return base + " The answer should be concise, precise, and natural."
    elif qt in ("multiple choice", "mcq"): return base + " The answer must be one of the provided options. Verify correctness in a natural, human-like manner."
    elif qt in ("true/false", "truefalse"): return base + " The answer must be either 'True' or 'False', explained in a simple, human-like way."
    else: return base + " The answer type is unspecified; check for correctness in a conversational style."

# Extract JSON string from OpenAI response
def extract_json_substring(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    return s[start:end+1] if start != -1 and end != -1 and end > start else s

# Make the OpenAI API call to check the validity of answers
def call_openai(question: str, answer: str, question_type: str) -> Dict[str, Any]:
    system_prompt = get_system_prompt(question_type)
    prompt = f"Question:\n{question}\n\nAnswer:\n{answer}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.8,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":prompt}],
            max_tokens=2000
        )
        raw = response.choices[0].message.content
        json_text = extract_json_substring(raw)
        try: result = json.loads(json_text)
        except json.JSONDecodeError: result = json5.loads(json_text)
        return result
    except openai.RateLimitError as e:
        error_msg = str(e)
        if 'insufficient_quota' in error_msg.lower() or 'quota' in error_msg.lower():
            raise QuotaExceededException(f"OpenAI API quota exceeded: {error_msg}")
        else:
            return {"valid": False, "reason": f"Rate limit error: {error_msg}", "corrected_answer": None}
    except openai.AuthenticationError as e:
        return {"valid": False, "reason": f"Authentication error: {str(e)}", "corrected_answer": None}
    except Exception as e:
        return {"valid": False, "reason": f"OpenAI API error: {str(e)}", "corrected_answer": None}

# Function to validate and process a record
def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    lang = "English"
    normalized_record = {normalize_key(k): v for k, v in record.items()}
    for possible_lang_key in FALLBACK_KEYS["language"]:
        if normalize_key(possible_lang_key) in normalized_record:
            lang = normalized_record[normalize_key(possible_lang_key)]
            break

    key_map = get_key_map_for_language(lang)
    mapped = map_record_fields(record, key_map)
    question = mapped.get("question", "")
    answer = mapped.get("answer", "")
    question_type = mapped.get("question_type", "unspecified")

    try:
        result = call_openai(question, answer, question_type)
        new_rec = record.copy()

        if result.get("valid", False):
            new_rec["_validation"] = {"valid": True, "reason": result.get("reason")}
            new_rec["explanation_status"] = "Answer is correct."
        else:
            # Replace the incorrect answer and explanation with the corrected ones
            if result.get("corrected_answer"):
                new_rec[key_map["answer"]] = result["corrected_answer"]
                new_rec["explanation_status"] = "Answer corrected by AI."
            if result.get("corrected_explanation"):
                new_rec[key_map["explanation"]] = result["corrected_explanation"]
            else:
                new_rec["explanation_status"] = "Explanation is already correct."
            new_rec["_validation"] = {"valid": False, "reason": result.get("reason"), "corrected": True}

        return new_rec
    except QuotaExceededException:
        # Re-raise quota exception to be caught at higher level
        raise

def gpt_translate_text(text_record: Dict[str, Any], target_language: str = "English") -> Dict[str, Any]:
    """
    Translate only the values of a JSON record to the target language.
    Keys remain in English exactly as specified in KEY_MAPS.
    """
    translated_record = {}
    for key, value in text_record.items():
        # Handle _validation metadata specially - translate the reason but keep structure
        if key == "_validation" and isinstance(value, dict):
            translated_validation = {}
            for val_key, val_value in value.items():
                if val_key == "reason" and isinstance(val_value, str) and val_value.strip():
                    # Translate the validation reason
                    try:
                        prompt = (
                            f"Translate the following validation reason into {target_language}. "
                            f"Keep technical terms and maintain the professional tone. "
                            f"Do not change formatting, numbers, or LaTeX.\n\n"
                            f"Text: {val_value}"
                        )
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            temperature=0.5,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1000
                        )
                        translated_reason = response.choices[0].message.content.strip()
                        translated_validation[val_key] = translated_reason
                    except openai.RateLimitError as e:
                        error_msg = str(e)
                        if 'insufficient_quota' in error_msg.lower() or 'quota' in error_msg.lower():
                            raise QuotaExceededException(f"Translation quota exceeded: {error_msg}")
                        else:
                            print(f"Rate limit error translating validation reason: {e}")
                            translated_validation[val_key] = val_value  # Keep original
                    except Exception as e:
                        print(f"Error translating validation reason: {e}")
                        translated_validation[val_key] = val_value  # Keep original
                else:
                    # Keep other validation fields as is (valid, corrected flags)
                    translated_validation[val_key] = val_value
            translated_record[key] = translated_validation
            continue

        # Keep explanation_status as is (it's already in English)
        if key == "explanation_status":
            translated_record[key] = value
            continue

        # If value is empty or non-string, keep as is
        if not isinstance(value, str) or value.strip() == "":
            translated_record[key] = value
            continue

        try:
            # Translate the value
            prompt = (
                f"Translate the following text into {target_language}. "
                f"Do not change formatting, numbers, or LaTeX. "
                f"Do not translate proper nouns unless needed.\n\n"
                f"Text: {value}"
            )
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            translated_text = response.choices[0].message.content.strip()
            translated_record[key] = translated_text
        except openai.RateLimitError as e:
            error_msg = str(e)
            if 'insufficient_quota' in error_msg.lower() or 'quota' in error_msg.lower():
                raise QuotaExceededException(f"Translation quota exceeded: {error_msg}")
            else:
                print(f"Rate limit error for key {key}: {e}")
                time.sleep(60)  # Wait 1 minute before continuing
                translated_record[key] = value
        except openai.AuthenticationError as e:
            print(f"Authentication error: {e}")
            raise QuotaExceededException(f"Authentication error during translation: {str(e)}")
        except Exception as e:
            print(f"Error translating key {key}: {e}")
            translated_record[key] = value

    return translated_record

# Process records in parallel to speed up validation
def process_records_parallel(records: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
    results = [None]*len(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_record, rec): i for i, rec in enumerate(records)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except QuotaExceededException:
                # Cancel remaining futures and re-raise
                for f in futures:
                    f.cancel()
                raise
    return results

# Ensure API key validity before continuing
if not check_api_key_validity():
    exit(1)  # Stop the program if API key is invalid