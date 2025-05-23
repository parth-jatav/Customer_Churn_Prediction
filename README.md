
<h1 align="center">📉 Customer Churn Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Enabled-success?style=flat-square&logo=python&logoColor=white&color=blue" />
  <img src="https://img.shields.io/badge/Model-XGBoost%20%26%20RandomForest-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-red?style=flat-square" />
</p>



> 🎯 A Machine Learning solution that predicts whether a customer is likely to leave a service, helping businesses reduce churn and maximize retention!



## 🧠 Project Summary

This project is focused on predicting **customer churn** using ML techniques. It includes:
- 🔍 Data exploration & cleaning  
- 🧪 Model training using **Random Forest** and **XGBoost**  
- ⚖️ Class balancing using **SMOTE**  
- 🎛️ Hyperparameter tuning with **GridSearchCV**  
- 📦 Model saving with **Pickle**  
- 🔮 Real-time prediction pipeline



## 📁 Project Structure

```
customer-churn-prediction/
│
├── data                                 # Raw and cleaned datasets
├── app.py                               # Main scipting code
├── templates/                           # Contains html code for rendering
├── Customer Churn Prediction.ipynb      # All the step of analysis in jupyter file
├── flaskapi.py                          # flask api code
├── README.md                            # Project documentation
└── webd.py                              # (Optional) streamlit code if want easy code for ui
```



## 🛠️ Tools & Tech Stack

| Category             | Tools & Libraries                                           |
|----------------------|-------------------------------------------------------------|
| 📊 Data Analysis      | Pandas, NumPy, Matplotlib, Seaborn                         |
| 🧠 ML Models          | Scikit-learn, XGBoost, Random Forest                       |
| ⚖️ Imbalance Handling | SMOTE from imbalanced-learn                                |
| 🧪 Model Tuning       | GridSearchCV                                               |
| 💾 Model Saving       | Pickle                                                     |
| 🧰 Deployment Ready    | Python Function Pipeline                                  |

---

<br>

## 🧪 Model Pipeline
```
A[Raw Data] --> B[Data Cleaning]
B --> C[EDA]
C --> D[Feature Encoding]
D --> E[SMOTE Balancing]
E --> F[Model Training (RF & XGB)]
F --> G[Evaluation & Tuning]
G --> H[Pickle Models & Encoder]
H --> I[Prediction Function]
```

<br>

## 🔍 How to Use

### 1️⃣ Clone the repository
```bash
git clone https://github.com/part-jatav/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```



### 3️⃣ Make predictions
```python
from churn_predictor import predict_churn

sample_input = {
  "gender": "Male",
  "SeniorCitizen": 0,
  ...
}
result = predict_churn(sample_input)
print("Churn Prediction:", result)
```


<br>

## 💡 Features

- ✅ Real-time churn prediction
- 📈 Optimized models with tuned hyperparameters
- ⚙️ Saved encoders for consistent prediction pipelines
- 🔗 Easy to plug into Streamlit or Flask apps

<br>

## 📸 Preview 

<p align="center">
  <img src="Homepage.png" width="500" />
</p>

<br>

## 🧑‍💻 About the Creator

**Aditya Jatav**  
Data Analyst | Full Stack Developer | ML Enthusiast  
🎓 B.Tech Civil Engineering | MNNIT Allahabad  
🌐 [Portfolio Website](https://aditya-jatav.netlify.app/)  
🔗 [LinkedIn](https://www.linkedin.com/in/aditya-jatav)



## 🚀 Future Improvements

-  Add logging & monitoring features  
-  Create an interactive dashboard  



## 🙌 Acknowledgements

Special thanks to:
- Scikit-learn & XGBoost communities  
- Imbalanced-learn library  
- Open-source contributors & mentors


> ⭐ Star this repo if you found it helpful!


