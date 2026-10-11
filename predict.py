from textSummarizer.pipeline.prediction import PredictionPipeline

if __name__ == "__main__":
    pipeline = PredictionPipeline()

    # Sample dialogue formatted like the Samsum dataset
    sample_dialogue = """
    Alex: Hey, are you heading to the lab this afternoon?
    Sam: Yes, I should be there around 4 PM after class.
    Alex: Could you return that textbook to the library desk for me? I won't make it to campus today.
    Sam: Sure, just leave it with the front desk or have someone hand it over before 3:30.
    Alex: Awesome, thank you so much!
    """

    print("--- Original Conversation ---")
    print(sample_dialogue.strip())

    print("\nGenerating summary...")
    summary = pipeline.predict(sample_dialogue)

    print("\n--- Fine-Tuned Model Summary ---")
    print(summary)