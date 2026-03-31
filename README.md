#  Crop Recommendation System
### Siva GangadharaRao Tadiboyina - 25BAI11508 - VIT Bhopal University

---

A machine learning-based system that recommends the most suitable crop to grow based on soil nutrients and environmental conditions. Built to help farmers make data-driven planting decisions and improve agricultural yield.

---

##  Problem Statement

Farmers in India often rely on intuition or tradition when choosing which crop to plant, without accounting for current soil conditions or weather patterns. This leads to poor yield, financial loss, and resource wastage. This project uses a classification ML model to recommend the best crop based on measurable soil and climate inputs.

---

##  Features

- Predicts the best crop from 22 possible options
- Takes soil nutrients (N, P, K), temperature, humidity, pH, and rainfall as inputs
- Returns the recommended crop along with a confidence score
- Trained on a clean, well-structured dataset with ~99% accuracy

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Pandas | Data loading and manipulation |
| Scikit-learn | ML model (Random Forest Classifier) |
| Matplotlib / Seaborn | Data visualization |
| Jupyter Notebook | Interactive development |

---

##  Project Structure

```
crop-recommendation/
│
├── crop_recommendation.ipynb   # Main Jupyter Notebook (EDA + Model)
├── model.py                    # Core ML model and predict function
├── recommend.py                # Main script (entry point)
├── Crop_recommendation.csv     # Dataset
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crop-recommendation.git
cd crop-recommendation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** should contain:
```
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

### 3. Download the Dataset

Download the **Crop Recommendation Dataset** from Kaggle:
 https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Place the CSV file in the project root as `Crop_recommendation.csv`.

---

##  How to Use

### Option A: Run the Script Directly

```bash
python recommend.py
```

### Option B: Call the Function in Your Code

```python
from model import recommend_crop

# Parameters: N, P, K, temperature, humidity, ph, rainfall
my_crop, confidence = recommend_crop(25, 55, 65, 18, 45, 7.5, 80)
print(f"Your crop: {my_crop} ({confidence:.1f}%)")
```

**Output:**
```
Your crop: wheat (91.0%)
```

### Option C: Jupyter Notebook

```bash
jupyter notebook crop_recommendation.ipynb
```

---

##  Input Parameters

| Parameter | Description | Unit | Example |
|-----------|-------------|------|---------|
| N | Nitrogen content in soil | mg/kg | 25 |
| P | Phosphorus content in soil | mg/kg | 55 |
| K | Potassium content in soil | mg/kg | 65 |
| Temperature | Average temperature | °C | 18 |
| Humidity | Relative humidity | % | 45 |
| pH | Soil pH value | 0–14 | 7.5 |
| Rainfall | Annual rainfall | mm | 80 |

---

##  Model Performance

- **Algorithm:** Random Forest Classifier
- **Accuracy:** ~99%
- **Dataset Size:** 2200 samples, 22 crop classes
- **Train/Test Split:** 80/20

---

##  Supported Crops

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

---

##  Author

**Siva GangadharaRao Tadiboyina**  
B.Tech Student | VIT  Bhopal University
Course: Fundamentals of AI and ML
