package com.example.navigationdrawer;

import android.content.Context;
import android.os.Bundle;
import androidx.fragment.app.Fragment;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;

import android.preference.PreferenceManager;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.content.SharedPreferences;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Switch;
import android.widget.TextView;

import java.util.concurrent.TimeUnit;

public class SettingsFragment extends Fragment {

    private RadioGroup radioGroupTheme;
    private Switch notificationSwitch;
    private SharedPreferences sharedPreferences,sharedPreferences2;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_settings, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        radioGroupTheme = view.findViewById(R.id.radio_group_theme);

        sharedPreferences = requireActivity().getSharedPreferences("AppPreferences", Context.MODE_PRIVATE);

        // Find the Switch view using the correct ID
        notificationSwitch = view.findViewById(R.id.switch_notifications);
        boolean isNotificationEnabled = sharedPreferences.getBoolean("notifications_enabled", true);
        notificationSwitch.setChecked(isNotificationEnabled);
        // Set the initial state of the switch based on stored preference

        TextView notificationStatusText = view.findViewById(R.id.text_notification_status);
        notificationStatusText.setText(isNotificationEnabled ? "ON" : "OFF");


        // Listener for the Switch
        notificationSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            sharedPreferences.edit().putBoolean("notifications_enabled", isChecked).apply();
            notificationStatusText.setText(isChecked ? "ON" : "OFF");

            // Start or stop feedback notification based on switch state
            if (isChecked) {
                startFeedbackNotification();
            } else {
                stopFeedbackNotification();
            }
        });


        Button resetButton = view.findViewById(R.id.button_reset);
        resetButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                resetToDefaultSettings(view);
            }
        });



        // Load saved theme preference
        sharedPreferences2 = requireActivity().getSharedPreferences("ThemePreferences", getContext().MODE_PRIVATE);
        boolean isDarkMode = sharedPreferences2.getBoolean("isDarkMode", false);
        setTheme(isDarkMode); // Apply the saved theme
        // Set initial selection based on saved preference
        if (isDarkMode) {
            ((RadioButton) view.findViewById(R.id.radio_dark)).setChecked(true);
        } else {
            ((RadioButton) view.findViewById(R.id.radio_light)).setChecked(true);
        }

        // Listener for theme selection
        radioGroupTheme.setOnCheckedChangeListener((group, checkedId) -> {
            SharedPreferences.Editor editor2 = sharedPreferences2.edit();
            boolean shouldRecreate = false;

            if (checkedId == R.id.radio_light && isDarkMode) {
                setTheme(false);
                editor2.putBoolean("isDarkMode", false);
                shouldRecreate = true;
            } else if (checkedId == R.id.radio_dark && !isDarkMode) {
                setTheme(true);
                editor2.putBoolean("isDarkMode", true);
                shouldRecreate = true;
            }

            editor2.apply();

            // Recreate only if theme has changed
            if (shouldRecreate) {
                requireActivity().recreate();
            }
        });


    }

    private void resetToDefaultSettings(View view){

        SharedPreferences.Editor editor = sharedPreferences.edit();
        SharedPreferences.Editor editor2 = sharedPreferences2.edit();

        // Reset theme to light mode
        editor2.putBoolean("isDarkMode", false);
        setTheme(false); // Apply light theme immediately

        // Reset notification switch to ON
        notificationSwitch.setChecked(true);
        editor.putBoolean("notifications_enabled", true);

        editor.apply();
        editor2.apply();

        // Update UI elements
        ((RadioButton) view.findViewById(R.id.radio_light)).setChecked(true);
        TextView notificationStatusText = view.findViewById(R.id.text_notification_status);
        notificationStatusText.setText("ON");

        // Start notifications if they were stopped
        startFeedbackNotification();
    }



    private void startFeedbackNotification() {
        PeriodicWorkRequest notificationRequest = new PeriodicWorkRequest.Builder(
                FeedbackNotificationWorker.class,
                20, TimeUnit.MINUTES) // Set to 20 minutes interval
                .build();
        WorkManager.getInstance(getContext())
                .enqueueUniquePeriodicWork("feedback_notification",
                        ExistingPeriodicWorkPolicy.REPLACE, notificationRequest);
    }

    private void stopFeedbackNotification() {
        WorkManager.getInstance(getContext()).cancelUniqueWork("feedback_notification");
    }


    private void setTheme(boolean darkMode) {
        if (darkMode) {
            requireActivity().setTheme(R.style.Theme_MyApp_Dark);
        } else {
            requireActivity().setTheme(R.style.Theme_MyApp_Light);
        }
    }
}