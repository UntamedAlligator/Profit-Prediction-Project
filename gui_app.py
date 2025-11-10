import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load trained model
try:
    model = joblib.load("profit_predictor.pkl")
except FileNotFoundError:
    messagebox.showerror("Error", "Trained model file 'profit_predictor.pkl' not found.")
    exit()

# Prediction function
def predict_profit():
    try:
        rd = float(entry_rd.get())
        admin = float(entry_admin.get())
        marketing = float(entry_marketing.get())

        features = np.array([[rd, admin, marketing]])
        profit = model.predict(features)[0]

        result_label.config(text=f"Predicted Profit: ₹{profit:,.2f}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values")

# Create GUI window
root = tk.Tk()
root.title("Profit Predictor")
root.geometry("300x300")
root.configure(bg="#f4f4f4")

tk.Label(root, text="R&D Spend:", bg="#f4f4f4").pack(pady=(10, 0))
entry_rd = tk.Entry(root)
entry_rd.pack()

tk.Label(root, text="Administration Spend:", bg="#f4f4f4").pack(pady=(10, 0))
entry_admin = tk.Entry(root)
entry_admin.pack()

tk.Label(root, text="Marketing Spend:", bg="#f4f4f4").pack(pady=(10, 0))
entry_marketing = tk.Entry(root)
entry_marketing.pack()

tk.Button(root, text="Predict Profit", command=predict_profit, bg="#4CAF50", fg="white").pack(pady=15)
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#f4f4f4")
result_label.pack()

root.mainloop()
