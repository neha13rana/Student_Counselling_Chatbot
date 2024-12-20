package com.example.navigationdrawer;

import java.util.ArrayList;
import java.util.List;

public class ChatSection {
    private long sessionId;
    private String title;
    private String timestamp;
    private List<String> messages;
    private boolean isExpanded;

    public ChatSection(long sessionId, String title, String timestamp) {
        this.sessionId = sessionId;
        this.title = title;
        this.timestamp = timestamp;
        this.messages = new ArrayList<>();
        this.isExpanded = false;  // Default to collapsed state
    }

    public String getTitle() {
        return title;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public List<String> getMessages() {
        return messages;
    }

    public boolean isExpanded() {
        return isExpanded;
    }

    public void setExpanded(boolean expanded) {
        isExpanded = expanded;
    }

    public void addMessage(String message, String sender) {
        messages.add(sender + ": " + message);  // Include sender for better readability
    }
}
