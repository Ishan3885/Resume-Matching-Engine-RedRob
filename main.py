import math

# Skill aliases
skill_aliases = {
    "pyhton": "python",
    "machinelearning": "machine_learning",
    "deeplearning": "deep_learning",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android",
    "firebase": "firebase",
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# Resume dataset
resumes = {
    "Arjun Sharma": ["Pyhton", "MachineLearning", "SQL", "pandas", "numpy", "Deep-learning"],
    "Priya Nair": ["JavaScrpit", "React", "Node.JS", "MongoDb", "REST api", "HTML/CSS"],
    "Rahul Gupta": ["Java", "Spring Boot", "MySql", "Microservices", "Docker", "kubernates"],
    "Sneha Patel": ["Python", "TensorFlow", "Keras", "NLP", "BERT", "data-viz", "matplotlib"],
    "Vikram Singh": ["C++", "Algoritms", "Data Structure", "competitive programming", "python"],
    "Ananya Krishnan": ["javascript", "vue.js", "python", "flask", "PostgreSQL", "AWS", "CI/CD"],
    "Karan Mehta": ["Python", "Sklearn", "XGboost", "feature engineering", "SQL", "tableau"],
    "Deepika Rao": ["Java", "Android", "Kotlin", "Firebase", "REST", "UI/UX", "figma"],
    "Aditya Kumar": ["Reactjs", "TypeScrpit", "GraphQL", "redux", "tailwind", "nodejs", "jest"],
    "Meera Iyer": ["python", "R", "statistics", "ML", "regression", "clustering", "Power-BI"],
}

# Job description dataset
job_descriptions = {
    "JD-1": {"company": "Kakao", "role": "ML Engineer", "required_skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "SQL", "Data Visualization"]},
    "JD-2": {"company": "Naver", "role": "Backend Engineer", "required_skills": ["Java", "Spring Boot", "MySQL", "PostgreSQL", "Microservices", "Docker", "Kubernetes"]},
    "JD-3": {"company": "Line", "role": "Frontend Engineer", "required_skills": ["JavaScript", "React", "Vue", "TypeScript", "REST API", "HTML/CSS"]},
}

def normalize_skills(skills):
    normalized_skills = []
    for skill in skills:
        # Convert to lowercase
        skill = skill.lower()
        
        # Apply alias mapping
        if skill in skill_aliases:
            skill = skill_aliases[skill]
        
        # Add to normalized skills
        normalized_skills.append(skill)
    
    # Remove duplicates
    normalized_skills = list(set(normalized_skills))
    
    return normalized_skills

def calculate_tf_idf(resumes):
    # Calculate TF-IDF for each resume
    tf_idf_vectors = {}
    vocabulary = set()
    for candidate, skills in resumes.items():
        normalized_skills = normalize_skills(skills)
        tf_idf_vector = {}
        for skill in normalized_skills:
            # Calculate TF
            tf = 1 / len(normalized_skills)
            
            # Calculate IDF
            idf = math.log(10 / sum(1 for skills in resumes.values() if skill in normalize_skills(skills)))
            
            # Calculate TF-IDF
            tf_idf = tf * idf
            
            # Add to TF-IDF vector
            tf_idf_vector[skill] = tf_idf
            
            # Add to vocabulary
            vocabulary.add(skill)
        
        tf_idf_vectors[candidate] = tf_idf_vector
    
    return tf_idf_vectors, list(vocabulary)

def create_binary_vector(jd_skills, vocabulary):
    binary_vector = [0] * len(vocabulary)
    for i, skill in enumerate(vocabulary):
        if skill in jd_skills:
            binary_vector[i] = 1
    
    return binary_vector

def calculate_cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vector1))
    magnitude2 = math.sqrt(sum(a ** 2 for a in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity

def rank_candidates(tf_idf_vectors, vocabulary, jd_skills):
    candidate_scores = {}
    binary_vector = create_binary_vector(jd_skills, vocabulary)
    for candidate, tf_idf_vector in tf_idf_vectors.items():
        # Create vector for candidate
        candidate_vector = [tf_idf_vector.get(skill, 0) for skill in vocabulary]
        
        # Calculate cosine similarity
        cosine_similarity = calculate_cosine_similarity(candidate_vector, binary_vector)
        
        # Add to candidate scores
        candidate_scores[candidate] = cosine_similarity
    
    # Sort candidates by score in descending order
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
    
    return sorted_candidates

def main():
    tf_idf_vectors, vocabulary = calculate_tf_idf(resumes)
    
    for jd, details in job_descriptions.items():
        jd_skills = normalize_skills(details["required_skills"])
        ranked_candidates = rank_candidates(tf_idf_vectors, vocabulary, jd_skills)
        
        print(f"{jd} — {details['company']} ({details['role']})")
        top_candidates = ranked_candidates[:3]
        for i, (candidate, score) in enumerate(top_candidates):
            if i < len(top_candidates) - 1:
                print(f"{candidate}({round(score, 2)})", end=", ")
            else:
                print(f"{candidate}({round(score, 2)})")
        print()

if __name__ == "__main__":
    main()
