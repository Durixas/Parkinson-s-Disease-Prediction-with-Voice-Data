import tkinter as tk
from tkinter import filedialog
import joblib
import os
import pandas as pd

# Initialize Tkinter application
root = tk.Tk()
root.title("Machine Learning Pipeline")

# Directory where models are saved
model_dir = 'saved_models'

model_name_mapping = {
    "Logistic Regression": "lr_model",
    "Decision Tree": "dt_model",
    "Random Forest": "rf_model",
    "Gradient Boosting": "gb_model",
    "XGBoost": "xgb_model",
    "LightGBM": "lgb_model",
    "K-Nearest Neighbors": "knn_model",
    "Support Vector Machine": "svm_model",
    "Naive Bayes": "nb_model",
    "AdaBoost": "ada_model",
    "MLP": "mlp_model",
    "Stacking": "stack_model"
}

# Function to load a model pipeline
def load_model(model_name):
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise ValueError(f"Model {model_name} is not recognized or does not exist.")

# Function to handle file upload
def upload_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    entry_path.delete(0, tk.END)
    entry_path.insert(0, filename)

# Function to handle prediction
def predict():
    filepath = entry_path.get()
    if not filepath:
        label_result.config(text="No file selected", fg="red")
        return

    if not os.path.exists(filepath):
        label_result.config(text="File does not exist", fg="red")
        return

    data = pd.read_csv(filepath)

    user_friendly_model_name = variable_model.get()
    model_name = model_name_mapping.get(user_friendly_model_name)

    pipeline = load_model(model_name)

    predictions = pipeline.predict(data)
    data['Predictions'] = predictions

    # Extract filename without extension
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = f"{file_name}_{model_name}.csv"

    data.to_csv(output_path, index=False)

    label_result.config(text=f"Predictions saved to {output_path}", fg="green")

# Create components
label_file = tk.Label(root, text="Select CSV File:")
label_file.grid(row=0, column=0, padx=10, pady=10)

entry_path = tk.Entry(root, width=50)
entry_path.grid(row=0, column=1, padx=10, pady=10)

button_browse = tk.Button(root, text="Browse", command=upload_file)
button_browse.grid(row=0, column=2, padx=10, pady=10)

label_model = tk.Label(root, text="Select Model:")
label_model.grid(row=1, column=0, padx=10, pady=10)

model_names = list(model_name_mapping.keys())
variable_model = tk.StringVar(root)
variable_model.set(model_names[0])
option_model = tk.OptionMenu(root, variable_model, *model_names)
option_model.grid(row=1, column=1, padx=10, pady=10)

button_predict = tk.Button(root, text="Predict", command=predict)
button_predict.grid(row=1, column=2, padx=10, pady=10)

label_result = tk.Label(root, text="", fg="green")
label_result.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Run the Tkinter application
root.mainloop()
