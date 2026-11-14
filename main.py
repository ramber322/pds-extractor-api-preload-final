from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from docx import Document
from openpyxl import load_workbook
import os
import re
import numpy as np
import joblib
from typing import List, Dict, Any, Optional
from datetime import datetime

app = FastAPI(
    title="AI-Enhanced PDS Screening API",
    version="3.0",
    description="Upload PDS and screen against LGU/CSC job requirements with relevance filtering"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

from feature_engine import (
    get_education_level,
    get_experience_years,
    get_training_hours,
    get_eligibility,
    detect_eligibility_level,
    get_eligibility_match_status,
    get_sentence_bert_model,
    extract_education_from_text,
    extract_eligibility_from_text,
    extract_training_hours_from_text,
    extract_experience_from_text,
)
print("Loading Sentence-BERT model...")
get_sentence_bert_model()

JOB_TITLE_TO_CATEGORY = {
    "administrative": ["administrative assistant", "administrative aide", "assessment clerk", "clerk", "secretary", "admin", "office staff"],
    "finance": ["budgeting assistant", "budget", "accountant", "accounting", "cashier", "treasury", "finance"],
    "engineering": ["engineer", "civil engineer", "mechanical engineer", "electrical engineer", "engineering"],
    "construction": ["building inspector", "labor foreman", "construction", "foreman", "inspector"],
    "communications": ["information officer", "public information", "media", "communications", "journalism", "writing"],
    "agriculture": ["slaughterhouse master", "agriculturist", "veterinary", "agriculture", "veterinarian"],
    "sports": ["sports officer", "recreation", "sports"],
    "technical": ["mechanic", "technician", "technical"],
    "it": ["it officer", "information technology", "it", "tech support", "developer"],
    "planning": ["planning officer", "development officer", "planning"],
    "social_services": ["social worker", "community affairs", "social", "community"]
}

DEFAULT_JOB_REQUIREMENTS = {
    "administrative": {"position_title": "Administrative Officer", "education": 3, "experience": 1, "training_hours": 4, "eligibility_level": "professional", "description": "Administrative support and office management roles"},
    "finance": {"position_title": "Budgeting Assistant / Accountant", "education": 3, "experience": 1, "training_hours": 4, "eligibility_level": "professional", "description": "Budgeting, accounting, procurement, and financial management roles"},
    "engineering": {"position_title": "Engineer III", "education": 3, "experience": 2, "training_hours": 8, "eligibility_level": "professional", "description": "Civil, mechanical, electrical engineering and design roles"},
    "construction": {"position_title": "Building Inspector / Foreman", "education": 3, "experience": 2, "training_hours": 8, "eligibility_level": "professional", "description": "Building inspection, construction supervision, and safety roles"},
    "communications": {"position_title": "Information Officer III", "education": 3, "experience": 2, "training_hours": 8, "eligibility_level": "professional", "description": "Public information, media relations, and communications roles"},
    "agriculture": {"position_title": "Slaughterhouse Master / Agriculturist", "education": 3, "experience": 2, "training_hours": 8, "eligibility_level": "professional", "description": "Agriculture, veterinary, food safety, and inspection roles"},
    "sports": {"position_title": "Sports Officer", "education": 3, "experience": 1, "training_hours": 4, "eligibility_level": "professional", "description": "Sports development, recreation, and physical education roles"},
    "technical": {"position_title": "Mechanic / Technician", "education": 2, "experience": 1, "training_hours": 4, "eligibility_level": "subprofessional", "description": "Mechanical, automotive, and equipment maintenance roles"},
    "it": {"position_title": "Information Technology Officer", "education": 3, "experience": 2, "training_hours": 8, "eligibility_level": "professional", "description": "IT, systems, networking, and software development roles"},
    "planning": {"position_title": "Planning Officer", "education": 3, "experience": 2, "training_hours": 8, "eligibility_level": "professional", "description": "Urban planning, development, and research roles"},
    "social_services": {"position_title": "Social Worker / Community Affairs Officer", "education": 3, "experience": 2, "training_hours": 8, "eligibility_level": "professional", "description": "Social work, community development, and welfare roles"},
}


def clean(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def is_noise_row(row: List[str]) -> bool:
    text = " ".join(row).lower()
    noise_patterns = [
        "personal data sheet", "warning:", "read the attached guide",
        "print legibly", "cs form", "page", "signature", "date",
        "wet signature", "continue on separate sheet", "do not abbreviate",
        "indicate n/a", "tick appropriate boxes", "please indicate country",
        "write in full", "dd/mm/yyyy", "if applicable", "separate sheet",
        "name of children", "photo", "right thumbmark", "subscribed and sworn",
        "i. personal information", "ii. family background", "iii. educational background",
        "iv. civil service", "v. work", "vi. voluntary", "vii. learning",
        "viii. other information", "ix.", "x.", "pursuant to", "references",
        "government issued id", "person administering oath", "affiant exhibiting",
        "indigenous people", "magna carta", "expanded solo parents",
        "candidate in a national", "resigned from the government",
        "immigrant or permanent resident", "status of appointment", "gov't service"
    ]
    for pattern in noise_patterns:
        if pattern in text:
            return True
    return len(text) < 3


def extract_docx(file_path: str) -> List[List[str]]:
    doc = Document(file_path)
    rows = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text and len(text) > 3:
            parts = re.split(r'\t{2,}|\s{3,}', text)
            cleaned = [clean(p) for p in parts if clean(p)]
            if cleaned and not is_noise_row(cleaned):
                rows.append(cleaned)
    return rows


def extract_xlsx(file_path: str) -> List[List[str]]:
    wb = load_workbook(file_path, data_only=False)
    rows = []
    for sheet_name in ["C1", "C2", "C3"]:
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        for row in ws.iter_rows(values_only=True):
            cleaned = [clean(c) for c in row if clean(c)]
            if cleaned:
                rows.append(cleaned)
    return rows


def normalize_pds(rows: List[List[str]]) -> Dict[str, List[List[str]]]:
    data = {"education": [], "experience": [], "eligibility": [], "training": []}
    in_education = in_experience = in_eligibility = in_training = False
    seen_education_header = seen_experience_header = seen_eligibility_header = seen_training_header = False

    for row in rows:
        if not row:
            continue
        text = " ".join(row).lower()

        if "educational background" in text or "iii. educational" in text:
            in_education, in_experience, in_eligibility, in_training = True, False, False, False
            seen_education_header = True
            continue
        if "work experience" in text or "v. work" in text:
            in_experience, in_education, in_eligibility, in_training = True, False, False, False
            seen_experience_header = True
            continue
        if "civil service" in text or "eligibility" in text or "cse" in text or "csee" in text:
            in_eligibility, in_education, in_experience, in_training = True, False, False, False
            seen_eligibility_header = True
            continue
        if "learning and development" in text or "training programs" in text or "vii. learning" in text:
            in_training, in_education, in_experience, in_eligibility = True, False, False, False
            seen_training_header = True
            continue

        if in_education and seen_education_header:
            if any(w in text for w in ["write in full", "period of attendance", "units earned", "level", "name of school"]):
                continue
            if len(row) >= 3 and "n/a" not in text:
                data["education"].append(row)
        elif in_experience and seen_experience_header:
            if any(w in text for w in [
                "write in full", "inclusive dates", "dd/mm/yyyy", "status of appointment",
                "agency employee", "employee no", "personal information",
                "surname", "first name", "middle name", "date of birth",
                "citizenship", "civil status", "residential address",
            ]):
                continue
            if len(row) >= 3:
                has_date = any(
                    isinstance(c, datetime) or re.search(r'(20\d{2}|19\d{2})', str(c)) or "present" in str(c).lower()
                    for c in row[:2]
                )
                if has_date:
                    data["experience"].append(row)
        elif in_eligibility and seen_eligibility_header:
            if any(w in text for w in ["rating", "date of examination", "license", "valid until", "ces/csee", "conferment"]):
                continue
            if len(row) >= 1:
                if all(not str(c).strip() or str(c).strip().lower() in ["n/a", "none", ""] for c in row if c is not None):
                    continue
                data["eligibility"].append(row)
        elif in_training and seen_training_header:
            if any(w in text for w in ["write in full", "inclusive dates", "number of hours", "type of l&d"]):
                continue
            if len(row) >= 3 and "n/a" not in text:
                data["training"].append(row)
    return data


def build_features(pds: Dict[str, List[List[str]]], job_category: str = "administrative") -> Dict[str, Any]:
    education_text = " ".join([" ".join([str(c) for c in row]) for row in pds["education"]])
    eligibility_text = " ".join([" ".join([str(c) for c in row]) for row in pds["eligibility"]])
    return {
        "education_level": get_education_level(education_text),
        "years_experience": get_experience_years(pds["experience"], job_category),
        "training_hours": get_training_hours(pds["training"], job_category),
        "eligibility": get_eligibility(eligibility_text),
        "eligibility_level": detect_eligibility_level(eligibility_text),
    }


def compute_score(applicant: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    breakdown = {}
    sum_of_progress = 0
    total_requirements = 0
    met_requirements = 0

    req_edu, val_edu = job.get("education", 1), applicant.get("education", 0)
    if req_edu == 0:
        mp, mb, status = 1.0, 1.0, "NOT REQUIRED"
    elif val_edu >= req_edu:
        mp, mb, status = 1.0, 1.0, "MET"
    else:
        mp = val_edu / req_edu
        mb = 0.0
        status = f"{round(mp * 100)}% (needs {req_edu - val_edu} more level)"
    breakdown["Education"] = {"required": req_edu, "actual": val_edu, "progress": round(mp, 2), "binary": mb, "status": status}
    if status != "NOT REQUIRED":
        total_requirements += 1
        sum_of_progress += mp
        if mb >= 1:
            met_requirements += 1

    req_exp, val_exp = job.get("experience", 1), applicant.get("experience", 0)
    if req_exp == 0:
        mp, mb, status = 1.0, 1.0, "NOT REQUIRED"
    elif val_exp >= req_exp:
        mp, mb, status = 1.0, 1.0, "MET"
    else:
        mp = val_exp / req_exp
        mb = 0.0
        status = f"{round(mp * 100)}% (needs {req_exp - val_exp} more years)"
    breakdown["Experience"] = {"required": req_exp, "actual": val_exp, "progress": round(mp, 2), "binary": mb, "status": status}
    if status != "NOT REQUIRED":
        total_requirements += 1
        sum_of_progress += mp
        if mb >= 1:
            met_requirements += 1

    req_train, val_train = job.get("training_hours", 1), applicant.get("training_hours", 0)
    if req_train == 0:
        mp, mb, status = 1.0, 1.0, "NOT REQUIRED"
    elif val_train >= req_train:
        mp, mb, status = 1.0, 1.0, "MET"
    else:
        mp = val_train / req_train if req_train > 0 else 1.0
        mb = 0.0
        status = f"{round(mp * 100)}% (needs {req_train - val_train} more hours)"
    breakdown["Training Hours"] = {"required": req_train, "actual": val_train, "progress": round(mp, 2), "binary": mb, "status": status}
    if status != "NOT REQUIRED":
        total_requirements += 1
        sum_of_progress += mp
        if mb >= 1:
            met_requirements += 1

    req_elig, app_elig = job.get("eligibility_level", "professional"), applicant.get("eligibility_level", "unknown")
    if req_elig == "none":
        mp, mb, status = 1.0, 1.0, "NOT REQUIRED"
    else:
        er = get_eligibility_match_status(app_elig, req_elig)
        mp = er["score"]
        mb = er["score"]
        status = f"{er['status']} ({er['message']})"
    breakdown["Eligibility"] = {"required": req_elig, "actual": app_elig, "progress": round(mp, 2), "binary": round(mb, 2), "status": status}
    if status != "NOT REQUIRED":
        total_requirements += 1
        sum_of_progress += mp
        if mb >= 1:
            met_requirements += 1

    if total_requirements == 0:
        hs, hp, bp, rm, verdict = 1.0, 100.0, 100.0, "0/0", "FIT"
    else:
        hs = sum_of_progress / total_requirements
        hp = round(hs * 100, 2)
        bp = round((met_requirements / total_requirements) * 100, 2)
        rm = f"{met_requirements}/{total_requirements}"
        verdict = "FIT" if met_requirements == total_requirements else "NOT FIT"

    return {
        "score": round(hs, 2),
        "percentage": hp,
        "binary_percentage": bp,
        "requirements_met": rm,
        "breakdown": breakdown,
        "verdict": verdict,
        "met_requirements": met_requirements,
        "total_requirements": total_requirements,
    }


def explain_prediction(applicant: Dict[str, Any], job: Dict[str, Any], score_result: Dict[str, Any], job_title: str = "") -> Dict[str, Any]:
    explanation = {"summary": "", "verdict_description": "", "job_title": job_title, "feature_breakdown": [], "recommendations": []}
    for feature, data in score_result["breakdown"].items():
        status = data["status"]
        actual = data["actual"]
        required = data["required"]
        progress = data.get("progress", 0)

        is_met = status == "MET" or "EXCEEDS" in status or ("MET" in status and "NOT" not in status) or ("MATCH" in status and "NO" not in status.upper())

        if "NOT REQUIRED" in status:
            desc = "Not required"
        elif is_met:
            desc = "Meets requirement"
            if feature == "Eligibility" and "(" in status and ")" in status:
                match = re.search(r'\((.*?)\)', status)
                if match:
                    desc = match.group(1)
                    if desc:
                        desc = desc[0].upper() + desc[1:] if len(desc) > 1 else desc
        else:
            desc = "Does not meet requirement"
            if feature == "Eligibility":
                explanation["recommendations"].append(f"Obtain the required eligibility: {required}")
            elif feature == "Education":
                gap = required - actual
                if gap > 0:
                    explanation["recommendations"].append(f"Complete {int(gap)} more education level(s)")
            elif feature == "Experience":
                gap = required - actual
                if gap > 0:
                    explanation["recommendations"].append(f"Gain {int(gap)} more years of relevant experience")
            elif feature == "Training Hours":
                gap = required - actual
                if gap > 0:
                    explanation["recommendations"].append(f"Complete {int(gap)} more hours of relevant training")

        explanation["feature_breakdown"].append({
            "feature": feature,
            "actual": actual,
            "required": required,
            "progress": round(progress * 100, 2),
            "status": status,
            "description": desc,
        })

    mc = score_result.get("met_requirements", 0)
    t = score_result.get("total_requirements", 0)

    if t == 0:
        explanation["summary"] = "No specific requirements for this position. You are eligible!"
    elif mc == t:
        explanation["summary"] = f"Applicant meets all {t} requirements! Perfect match."
    elif mc >= t * 0.75:
        explanation["summary"] = f"Applicant meets {mc} out of {t} requirements. Strong candidate with minor gaps."
    elif mc > 0:
        explanation["summary"] = f"Applicant meets only {mc} out of {t} requirements. Needs improvement."
    else:
        explanation["summary"] = "Applicant meets NO requirements."

    explanation["verdict_description"] = (
        "Qualified and recommended for further consideration." if score_result["verdict"] == "FIT"
        else "Does not meet all requirements. Review recommendations above."
    )
    return explanation


def ml_predict(features: Dict[str, Any]) -> Optional[int]:
    if not model:
        return None
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    X = np.array([[features["education_level"], features["years_experience"], features["training_hours"], features["eligibility"]]])
    return int(model.predict(X)[0])


def detect_job_category(position_title: str) -> str:
    title_lower = position_title.lower()
    for category, keywords in JOB_TITLE_TO_CATEGORY.items():
        for keyword in keywords:
            if keyword in title_lower:
                return category
    return "administrative"


@app.post("/upload-pds")
async def upload_pds(
    file: UploadFile = File(..., description="PDS file (Excel, Word, or PDF)"),
    position_title: Optional[str] = Query(None, description="Position title"),
    job_category: Optional[str] = Query(None, description="Job category"),
    education: Optional[int] = Query(None, description="Required education level (0-5)"),
    experience: Optional[int] = Query(None, description="Required years of experience"),
    training_hours: Optional[int] = Query(None, description="Required training hours"),
    eligibility_level: Optional[str] = Query(None, description="Required eligibility"),
):
    temp_file = f"temp_{file.filename}"

    try:
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        filename_lower = file.filename.lower()
        is_pdf = False
        pdf_text = None
        raw = None

        if filename_lower.endswith(".docx"):
            raw = extract_docx(temp_file)
        elif filename_lower.endswith((".xlsx", ".xls")):
            raw = extract_xlsx(temp_file)
        elif filename_lower.endswith(".pdf"):
            is_pdf = True
            try:
                from pds_parser import extract_pdf
                pdf_text = extract_pdf(temp_file)
                if not pdf_text or len(pdf_text.strip()) < 100:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Could not extract readable text from the PDF. Please upload a clearer scan."}
                    )
                    # === TEMP DEBUG: dump the raw training section ===
                print("\n" + "=" * 70)
                print("RAW PDF TRAINING SECTION")
                print("=" * 70)
                _idx = pdf_text.upper().find("LEARNING AND DEVELOPMENT")
                if _idx >= 0:
                    print(pdf_text[_idx:_idx + 2000])
                else:
                    print("SECTION NOT FOUND — 'LEARNING AND DEVELOPMENT' missing from PDF text")
                print("=" * 70 + "\n")
                # === END TEMP DEBUG ===
            except Exception as pdf_err:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to read PDF: {str(pdf_err)}"}
                )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format. Please upload .docx, .xlsx, .xls, or .pdf"}
            )

        if job_category:
            detected_category = job_category.lower()
        elif position_title:
            detected_category = detect_job_category(position_title)
        else:
            detected_category = "administrative"

        if detected_category not in DEFAULT_JOB_REQUIREMENTS:
            detected_category = "administrative"

        defaults = DEFAULT_JOB_REQUIREMENTS[detected_category]

        final_education = education if education is not None else defaults["education"]
        final_experience = experience if experience is not None else defaults["experience"]
        final_training = training_hours if training_hours is not None else defaults["training_hours"]
        final_eligibility = eligibility_level if eligibility_level is not None else defaults["eligibility_level"]

        job = {
            "education": final_education,
            "experience": final_experience,
            "training_hours": final_training,
            "eligibility_level": final_eligibility,
        }

        if is_pdf:
            education_val = extract_education_from_text(pdf_text)
            eligibility_val = extract_eligibility_from_text(pdf_text)
            training_val = extract_training_hours_from_text(pdf_text, detected_category)
            experience_val = extract_experience_from_text(pdf_text)

            applicant = {
                "education": education_val,
                "experience": experience_val,
                "training_hours": training_val,
                "eligibility": 1 if eligibility_val != "unknown" else 0,
                "eligibility_level": eligibility_val,
            }

            low_confidence = []
            if education_val == 0:
                low_confidence.append("education")
            if eligibility_val == "unknown":
                low_confidence.append("eligibility")
            if experience_val == 0:
                low_confidence.append("experience")
            if training_val == 0:
                low_confidence.append("training_hours")

            score_result = compute_score(applicant, job)
            job_title_display = position_title if position_title else defaults.get("position_title", "Position")
            explanation = explain_prediction(applicant, job, score_result, job_title_display)

            return {
                "success": True,
                "filename": file.filename,
                "source_format": "pdf",
                "position_title": job_title_display,
                "job_category": detected_category,
                "job_description": defaults.get("description", ""),
                "applicant_features": applicant,
                "job_requirements": job,
                "fit_analysis": score_result,
                "explanation": explanation,
                "requires_manual_review": len(low_confidence) > 0,
                "low_confidence_fields": low_confidence,
                "extraction_note": (
                    f"Could not reliably extract: {', '.join(low_confidence)}. Verify manually."
                    if low_confidence else "All fields extracted from PDF."
                ),
                "ml_prediction": None,
                "raw_data_summary": {"source": "pdf_ocr", "text_length": len(pdf_text)},
            }

        if not raw:
            return JSONResponse(
                status_code=400,
                content={"error": "No data could be extracted from the file"}
            )

        pds = normalize_pds(raw)
        features = build_features(pds, detected_category)

        applicant = {
            "education": features["education_level"],
            "experience": features["years_experience"],
            "training_hours": features["training_hours"],
            "eligibility": features["eligibility"],
            "eligibility_level": features["eligibility_level"],
        }

        score_result = compute_score(applicant, job)
        job_title_display = position_title if position_title else defaults.get("position_title", "Position")
        explanation = explain_prediction(applicant, job, score_result, job_title_display)
        ml_result = ml_predict(features)

        return {
            "success": True,
            "filename": file.filename,
            "source_format": "excel",
            "position_title": job_title_display,
            "job_category": detected_category,
            "job_description": defaults.get("description", ""),
            "applicant_features": applicant,
            "job_requirements": job,
            "fit_analysis": score_result,
            "explanation": explanation,
            "ml_prediction": ml_result,
            "requires_manual_review": False,
            "low_confidence_fields": [],
            "raw_data_summary": {
                "education_entries": len(pds["education"]),
                "experience_entries": len(pds["experience"]),
                "eligibility_entries": len(pds["eligibility"]),
                "training_entries": len(pds["training"]),
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


@app.get("/eligibility-options")
async def get_eligibility_options():
    return {
        "eligibility_options": [
            {"value": "professional", "label": "Career Service (Professional) Second Level"},
            {"value": "subprofessional", "label": "Career Service (Subprofessional) First Level"},
            {"value": "none", "label": "None Required"},
        ]
    }


@app.get("/job-categories")
async def get_job_categories():
    result = {}
    for category, data in DEFAULT_JOB_REQUIREMENTS.items():
        result[category] = {
            "position_title": data["position_title"],
            "default_requirements": {
                "education": data["education"],
                "experience": data["experience"],
                "training_hours": data["training_hours"],
                "eligibility_level": data["eligibility_level"],
            },
            "description": data["description"],
        }
    return {"categories": result}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}