package com.example.navigationdrawer;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import androidx.annotation.NonNull;
import androidx.core.app.NotificationCompat;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

public class FeedbackNotificationWorker extends Worker {

    public FeedbackNotificationWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @NonNull
    @Override
    public Result doWork() {
        sendFeedbackNotification();
        return Result.success();
    }

    private void sendFeedbackNotification() {
        String channelId = "feedback_channel";
        String channelName = "Feedback Notification";
        NotificationManager notificationManager = (NotificationManager) getApplicationContext().getSystemService(Context.NOTIFICATION_SERVICE);

        // Create the NotificationChannel, required for Android 8.0 and higher
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_DEFAULT);
            notificationManager.createNotificationChannel(channel);
        }

        // Build the notification
        Notification notification = new NotificationCompat.Builder(getApplicationContext(), channelId)
                .setContentTitle("Feedback Request!")
                .setContentText("We request you to provide your feedback for our Chatbot.")
                .setSmallIcon(R.drawable.notification) // Replace with your app's icon
                .build();

        // Display the notification
        notificationManager.notify(1, notification);
    }
}
