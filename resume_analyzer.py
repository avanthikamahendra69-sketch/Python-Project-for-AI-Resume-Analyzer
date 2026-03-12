from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# System Introduction
print("\n======================================")
print(" AI Resume Screening System")
print(" Deloitte Graduate Hiring Assessment")
print("======================================\n")

print("Loading resumes...")
print("Analyzing candidate skills...")
print("Matching resumes with job description...\n")

# Job description
job_description = """
Looking for a Python developer with skills in machine learning,
data analysis, NLP, and artificial intelligence.
"""

print("Job Description:\n")
print(job_description)

# Skills to extract
skills = [
    "python",
    "machine learning",
    "data analysis",
    "nlp",
    "deep learning",
    "tensorflow"
]

# Load resumes
with open("resumes.txt", "r") as file:
    resumes = file.readlines()

documents = [job_description] + resumes

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(documents)

# Similarity calculation
similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

# Combine resumes with scores
candidates = list(zip(resumes, similarity_scores))

# Rank candidates
ranked_candidates = sorted(
    candidates,
    key=lambda x: x[1],
    reverse=True
)

print("\n======================================")
print(" AI Resume Screening Results")
print("======================================")

for i, (resume, score) in enumerate(ranked_candidates, start=1):

    print(f"\nCandidate Rank: {i}")
    print(f"Similarity Score: {score:.2f}")
    print("Resume:", resume)

    extracted_skills = [skill for skill in skills if skill in resume.lower()]

    skill_match = (len(extracted_skills) / len(skills)) * 100

    print("Extracted Skills:", extracted_skills)
    print("Skill Match Percentage:", round(skill_match, 2), "%")

# Hiring Recommendation
best_candidate = ranked_candidates[0]

print("\n======================================")
print(" AI Hiring Recommendation")
print("======================================")
print("Best Candidate Resume:")
print(best_candidate[0])

# Top 3 candidates
print("\n======================================")
print(" Top 3 Recommended Candidates")
print("======================================")

for i in range(3):
    print(f"\nCandidate {i+1}:")
    print(ranked_candidates[i][0])

# Recruiter Summary
print("\n======================================")
print(" Recruiter Summary Report")
print("======================================")

print("Total Resumes Processed:", len(resumes))
print("Top Candidates Identified Successfully")

print("\n======================================")
print(" AI Screening Completed Successfully")
print(" Top candidates recommended to recruiter")
print("======================================")
