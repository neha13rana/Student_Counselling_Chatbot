package com.example.navigationdrawer;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RatingBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class FeedBackFragment extends Fragment {
    private EditText feedbackEditText;
    private RatingBar ratingBar;
    private Button submitButton, clearButton;
    private TextView feedbackDisplayTextView;

    private SharedPreferences sharedPreferences;
    private static final String PREFS_NAME = "FeedbackPrefs";
    private static final String KEY_FEEDBACK = "feedback";
    private static final String KEY_RATING = "rating";

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view = inflater.inflate(R.layout.fragment_feed_back, container, false);

        feedbackEditText = view.findViewById(R.id.feedback_edit_text);
        ratingBar = view.findViewById(R.id.rating_bar);
        submitButton = view.findViewById(R.id.submit_button);
        clearButton = view.findViewById(R.id.clear_feedback_button);
        feedbackDisplayTextView = view.findViewById(R.id.feedback_display_text_view);

        sharedPreferences = requireContext().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);

        loadFeedback();

        submitButton.setOnClickListener(v -> handleFeedbackSubmission());
        clearButton.setOnClickListener(v -> clearFeedback());

        return view;
    }

    private void loadFeedback() {
        String savedFeedback = sharedPreferences.getString(KEY_FEEDBACK, null);
        float savedRating = sharedPreferences.getFloat(KEY_RATING, 0);

        if (savedFeedback != null) {
            feedbackEditText.setText(savedFeedback);
            ratingBar.setRating(savedRating);
            submitButton.setText("Edit");
            feedbackDisplayTextView.setText("Your Feedback: " + savedFeedback + "\nRating: " + savedRating);
        } else {
            feedbackDisplayTextView.setText("");
            submitButton.setText("Submit");
        }
    }

    private void handleFeedbackSubmission() {
        String feedback = feedbackEditText.getText().toString().trim();
        float rating = ratingBar.getRating();

        // Check if feedback is empty or rating is zero
        if (feedback.isEmpty()) {
            Toast.makeText(getContext(), "Please enter feedback before submitting.", Toast.LENGTH_SHORT).show();
            return;
        } else if (rating == 0) {
            Toast.makeText(getContext(), "Please provide a rating before submitting.", Toast.LENGTH_SHORT).show();
            return;
        }

        // Save feedback and rating to SharedPreferences
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString(KEY_FEEDBACK, feedback);
        editor.putFloat(KEY_RATING, rating);
        editor.apply();

        // Display the feedback and rating below and reset the input fields
        feedbackDisplayTextView.setText("Your Feedback: " + feedback + "\nRating: " + rating);
        feedbackEditText.setText(""); // Clear the EditText
        ratingBar.setRating(0); // Reset the RatingBar to default

        Toast.makeText(getContext(), "Feedback saved successfully!", Toast.LENGTH_SHORT).show();
        submitButton.setText("Edit"); // Change button text to "Edit"
    }

    private void clearFeedback() {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.remove(KEY_FEEDBACK);
        editor.remove(KEY_RATING);
        editor.apply();

        feedbackEditText.setText(""); // Clear the EditText
        ratingBar.setRating(0); // Reset the RatingBar to default
        submitButton.setText("Submit"); // Change button text back to "Submit"
        feedbackDisplayTextView.setText(""); // Clear the feedback display

        Toast.makeText(getContext(), "Feedback cleared.", Toast.LENGTH_SHORT).show();
    }
}
