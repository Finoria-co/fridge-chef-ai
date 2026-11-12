# Person 5 — Core Recipe ML

## ไฟล์ในโฟลเดอร์นี้

### recipe_model.ipynb
Training code ทั้งหมด ประกอบด้วย:
- โหลด recipes_dataset.csv จาก Person 3
- สร้าง binary feature matrix (12 ingredients)
- Train 3 โมเดล: TF-IDF + Cosine Similarity, KNN, Random Forest
- เปรียบเทียบ accuracy ของแต่ละโมเดล
- บันทึกโมเดลเป็น recipe_model.pkl

### recipe_model.pkl
โมเดลที่ train เสร็จแล้ว พร้อมใช้งาน ประกอบด้วย:
- vectorizer, tfidf_matrix
- knn_model, rf_model
- df (DataFrame 50 สูตร)
- ingredient_keywords

### recipe_utils.py
Wrapper function สำหรับ Person 6 ใช้งานได้เลย:
- load_model(path) — โหลด .pkl
- recommend(user_ingredients, bundle, model, top_n) — แนะนำสูตรอาหาร

## วิธีใช้
from recipe_utils import load_model, recommend
bundle = load_model('recipe_model.pkl')
results = recommend(['หมูสับ', 'ไข่'], bundle, model='knn')

## ผลลัพธ์ที่ได้
- name_th, name_en, cuisine
- match_pct — เปอร์เซ็นต์ความตรง
- have — วัตถุดิบที่มีแล้ว
- missing — วัตถุดิบที่ขาด
- description, cooking_method, difficulty, calories_kcal
