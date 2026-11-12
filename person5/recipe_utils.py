import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_model(path='recipe_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def recommend(user_ingredients, bundle, model='knn', top_n=5):
    df                 = bundle['df']
    ingredient_keywords = bundle['ingredient_keywords']
    vectorizer         = bundle['vectorizer']
    tfidf_matrix       = bundle['tfidf_matrix']
    knn_model          = bundle['knn_model']
    rf_model           = bundle['rf_model']

    # สร้าง user vector (binary)
    user_vector = [
        1 if any(
            kw in ing
            for ing in user_ingredients
            for kw in kws
            if not (ing.strip().startswith('ไข่') and kw == 'ไก่')
        )
        else 0
        for kws in ingredient_keywords.values()
    ]

    if model == 'knn':
        distances, indices = knn_model.kneighbors([user_vector], n_neighbors=top_n)
        top_indices = indices[0]
    elif model == 'tfidf':
        user_str    = ';'.join(user_ingredients)
        user_vec    = vectorizer.transform([user_str])
        scores      = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
    elif model == 'rf':
        proba       = rf_model.predict_proba([user_vector])[0]
        top_indices = proba.argsort()[::-1][:top_n]

    col_to_name = {
        'has_pork': 'หมู', 'has_chicken': 'ไก่', 'has_shrimp': 'กุ้ง',
        'has_bacon': 'เบคอน', 'has_egg': 'ไข่', 'has_rice': 'ข้าว',
        'has_pasta': 'พาสต้า', 'has_bread': 'ขนมปัง', 'has_holy_basil': 'กะเพรา',
        'has_tomato': 'มะเขือเทศ', 'has_cheese': 'ชีส', 'has_cabbage': 'ผักกาดขาว'
    }

    results = []
    for idx in top_indices:
        row          = df.iloc[idx]
        recipe_needs = [c for c in ingredient_keywords if row[c] == 1]
        user_has     = [c for c, kws in ingredient_keywords.items()
                        if any(
                            kw in ing
                            for ing in user_ingredients
                            for kw in kws
                            if not (ing.strip().startswith('ไข่') and kw == 'ไก่')
                        )]
        have         = [col_to_name[c] for c in recipe_needs if c in user_has]
        missing      = [col_to_name[c] for c in recipe_needs if c not in user_has]
        match_pct    = round(len(have) / len(recipe_needs) * 100, 1) if recipe_needs else 0

        results.append({
            'recipe_id': row['recipe_id'],
            'name_th':   row['name_th'],
            'name_en':   row['name_en'],
            'cuisine':   row['cuisine'],
            'match_pct': match_pct,
            'have':      have,
            'missing':   missing,
            'description':    row['description'],
            'cooking_method': row['cooking_method'],
            'prep_time_min':  int(row['prep_time_min']),
            'cook_time_min':  int(row['cook_time_min']),
            'difficulty':     row['difficulty'],
            'calories_kcal':  int(row['calories_kcal']),
            'allergens':      row['allergens'],
        })

    return results

