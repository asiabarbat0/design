# In /Users/asiabarbato/Downloads/designstreamaigrok/tests/test_clip_embeddings.py
import numpy as np
from scipy.spatial.distance import cosine
import requests
import pytest
from app.services.recommender import _embed_image

def test_random_vector_distance():
    random_vector = np.random.randn(512)
    random_vector = random_vector / np.linalg.norm(random_vector)

    sample_url = 'https://images.unsplash.com/photo-1600585154340-be6161a56a0c'
    try:
        params = {'room_photo_url': sample_url}
        response = requests.get('http://localhost:5001/widget/recommendations', params=params)  # Changed to port 5001
        response.raise_for_status()
        data = response.json()

        recommendations = data.get('recommendations', [])
        assert recommendations, "No recommendations returned from endpoint"

        query_embedding = _embed_image(sample_url)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        similarities = []
        for i, item in enumerate(recommendations):
            similarity = 1 - cosine(random_vector, query_embedding)
            similarities.append(similarity)
            print(f"Cosine similarity with recommendation {i+1} (variant_id: {item['variant_id']}): {similarity:.4f}")

        assert all(0 <= sim <= 1 for sim in similarities), "Invalid similarity values"
        assert len(similarities) > 0, "No similarities calculated"

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Error calling endpoint: {e}")

if __name__ == "__main__":
    test_random_vector_distance()