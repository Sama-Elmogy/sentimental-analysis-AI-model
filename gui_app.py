import customtkinter as ctk
from joblib import load

# Load the trained model and vectorizer
model = load("sentiment_model.pkl")  # does the calculations to predict either the input is +ve or -ve
vectorizer = load("vectorizer.pkl")  # simply we can consider it as the dictionary for the model to understand the input

# Function for predicting sentiment
def predict_sentiment(review):
    transformed_review = vectorizer.transform([review])
    sentiment_label = model.predict(transformed_review)[0]  # [0] is for taking list first "and only" element
    confidence = max(model.predict_proba(transformed_review)[0])  # 2d list, getting the first "and only" row of values
    # "+ve" and "-ve"
    return sentiment_label, confidence

# GUI using customtkinter
def create_gui():
    # Initialize the customtkinter application
    ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

    root = ctk.CTk()
    root.title("Sentimental Analyzer")
    root.geometry("500x500")

    # Title Label
    title_label = ctk.CTkLabel(root, text="Sentimental Analyzer", font=("Arial", 24, "bold"))
    title_label.pack(pady=20)

    # Text Input
    review_label = ctk.CTkLabel(root, text="Enter your review:", font=("Arial", 12))
    review_label.pack(pady=10)

    # Entry
    review_entry = ctk.CTkTextbox(root, height=100, width=400)
    review_entry.pack(pady=10)

    # Indicator Box
    indicator_box = ctk.CTkFrame(root, width=50, height=50, corner_radius=10)
    indicator_box.pack(pady=10)

    # Output frame
    output_frame = ctk.CTkFrame(root, height=100, width=400, fg_color="white")
    output_frame.pack(pady=10)

    # Output Area
    output_label = ctk.CTkLabel(output_frame, text="", font=("Arial", 20))
    output_label.pack(pady=10)

    # Function to analyze sentiment
    def analyze_review():
        review_text = review_entry.get("1.0", "end").strip()
        if not review_text:
            output_label.configure(text="Please enter a review!")
            indicator_box.configure(fg_color="gray")
            return

        sentiment, confidence = predict_sentiment(review_text)
        output_label.configure(text=f"Sentiment: {sentiment}\nConfidence: {confidence:.2f}")

        # Update the indicator box color
        if sentiment == "positive":
            indicator_box.configure(fg_color="green")
        else:
            indicator_box.configure(fg_color="red")

    # Function to reset inputs and outputs
    def reset_():
        review_entry.delete("1.0", "end")
        output_label.configure(text="")
        indicator_box.configure(fg_color="gray")

    # Function to copy result to clipboard
    def copy_result():
        root.clipboard_clear()
        root.clipboard_append(output_label.cget("text"))
        root.update()

    # Function to save report to a text file
    def save_report():
        review_text = review_entry.get("1.0", "end").strip()
        sentiment_confidence_text = output_label.cget("text")
        with open("Save_Reports.txt", "a") as report_file:
            report_file.write(f"Review: {review_text}\n{sentiment_confidence_text}\n{'-' * 40}\n")

    # Buttons Frame
    buttons_frame = ctk.CTkFrame(root)
    buttons_frame.pack(pady=10)

    # Analyze Button
    analyze_button = ctk.CTkButton(buttons_frame, text="Analyze Sentiment", command=analyze_review, fg_color="#3d85c6")
    analyze_button.grid(row=0, column=0, padx=10)

    # Reset Button
    reset_button = ctk.CTkButton(buttons_frame, text="Reset", command=reset_, fg_color="#E63946")
    reset_button.grid(row=0, column=1, padx=10)

    # Copy Button
    copy_button = ctk.CTkButton(buttons_frame, text="Copy Result", command=copy_result, fg_color="#457B9D")
    copy_button.grid(row=0, column=2, padx=10)

    # Save Button
    save_button = ctk.CTkButton(buttons_frame, text="Save Report", command=save_report, fg_color="#2A9D8F")
    save_button.grid(row=0, column=3, padx=10)

    # Run the application
    root.mainloop()

# main function
create_gui()