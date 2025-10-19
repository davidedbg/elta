Titanic Survival Classifier

A simple deep-learning project that predicts passenger survival on the Kaggle Titanic dataset.
The project includes:

train.py — trains and saves a PyTorch model

app.py — Streamlit app for inference and evaluation

**Setup & Installation**

# Clone the repository
git clone https://github.com/davidedbg/titanic-classifier.git

# Install dependencies
pip install torch torchvision torchaudio pandas numpy matplotlib seaborn scikit-learn streamlit opendatasets

**Run Instructions**
# Train the Model
python train.py

This will:

Download and preprocess the Titanic dataset

Train a feed-forward neural network for 30 epochs

Save the trained weights as model_weights.pth

**Launch the Streamlit App**
streamlit run app.py
Then open the local URL (e.g., http://localhost:8501) to access the interface.

**Example Usage**
Upload a test CSV file (with a Survived column).

The app loads your saved model and preprocesses the data.

You’ll see:

Accuracy, recall, f1_score and classification report

Confusion matrix plot

Histogram of predicted probabilities

**Example Screenshots:**
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/956060e1-69c0-42e0-b751-467f6d474d07" />

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/373b43bb-0796-4c0a-9c85-18f8c9f1b18e" />


<img width="1005" height="865" alt="image" src="https://github.com/user-attachments/assets/93099de3-a1df-484b-bb7b-61adedc1013a" />


<img width="1125" height="865" alt="image" src="https://github.com/user-attachments/assets/78382e77-6846-4c2e-8ab1-11e78da0703f" />


**Model Architecture & Design**
Preprocessing:
* drop columns that are irrelevent to the task
* Numerical features → imputed (median) + MinMaxScaler
* Categorical features → imputed (most frequent) + OneHotEncoder
  
Model:
Input (12 features) → Linear(12→8) → ReLU
Linear(8→4) → ReLU
Linear(4→1) → Sigmoid
Loss: Binary Cross-Entropy
# optional change 
# adding the line: loss = loss + (1-abs(output - 0.5) + (1- output)).mean() that penalize uncertainty and values close to zero.
# it gives lower overall accuracy but better balance between "0" label and "1" label
# as can be seen in the prediction distribution and confusion matrix made by streamlit
Optimizer: Adam
Evaluation: Accuracy, classification report, confusion matrix
