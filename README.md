# ⭐ Music Recommendation System using SVD (Last.fm Dataset)

This project builds a **Music Recommender System** using **Matrix Factorization (SVD)** from the `surprise` library.
It uses the **Last.fm Dataset**, applies preprocessing to generate user–track play counts, trains an SVD model, evaluates it using RMSE, and finally recommends songs to any user.

---

## 📌 Features

* ✔ Loads & preprocesses the Last.fm dataset
* ✔ Aggregates **play counts** for (Username × Track)
* ✔ Builds a **Collaborative Filtering model using SVD**
* ✔ Splits data into train/test sets
* ✔ Evaluates model using **RMSE**
* ✔ Provides **Top-N song recommendations** for any user

---

## 📂 Project Structure

```
├── Last.fm_data.csv
├── recommender.py (or notebook)
├── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn scikit-surprise
```

---

## 📥 Load & Preprocess Data

The dataset is loaded and grouped to generate play counts:

```python
def load_data():
    df = pd.read_csv('/content/Last.fm_data.csv')
    df = df.head(50000)
    play_counts = df.groupby(['Username', 'Track']).size().reset_index(name='Playcount')
    return play_counts
```

---

## 🤖 Model Training (SVD)

The model is trained using Surprise’s SVD algorithm:

```python
reader = Reader(rating_scale=(0, data['Playcount'].max()))
dataset = Dataset.load_from_df(data[['Username', 'Track', 'Playcount']], reader)
trainset, testset = train_test_split(dataset, test_size=0.2)
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
```

Performance is measured using **RMSE**.

---

## 🎵 Making Recommendations

The system predicts ratings for songs a user has not listened to and returns the top-N recommendations:

```python
recommend_songs(user_id, n=5)
```

---

## 🧪 Example Output

```
🎵 Top 5 Recommended Songs for User: user_123
1. Song_A (Predicted Rating: 4.52)
2. Song_B (Predicted Rating: 4.48)
3. Song_C (Predicted Rating: 4.37)
...
```

---

## 📈 Model Evaluation

The model prints the **RMSE score**, indicating how well it predicts play counts.
Lower RMSE = better performance.

---

## 🛠 Technologies Used

* **Python**
* **Pandas**
* **Scikit-Surprise**
* **NumPy**
* **Machine Learning (Collaborative Filtering)**

---

## 📌 Future Improvements

* Add user-based & item-based KNN models
* Add content-based filtering (song metadata)
* Build a Streamlit web UI
* Optimize SVD using grid search
* Add emotion-based recommendations

---

## 🙌 Acknowledgements

Dataset: Last.fm User Listening Data
Recommender Algorithm: Surprise SVD

