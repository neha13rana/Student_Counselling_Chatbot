package com.example.navigationdrawer;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class ChatDatabaseHelper extends SQLiteOpenHelper {
    private static final String DATABASE_NAME = "chat_history.db";
    private static final int DATABASE_VERSION = 3;
    public long currentSessionId = -1;

    public ChatDatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String CREATE_SESSIONS_TABLE = "CREATE TABLE chat_sessions (" +
                "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "title TEXT, " +
                "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)";
        db.execSQL(CREATE_SESSIONS_TABLE);

        String CREATE_MESSAGES_TABLE = "CREATE TABLE chat_messages (" +
                "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "session_id INTEGER, " +
                "message TEXT, " +
                "sender TEXT, " +
                "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, " +
                "FOREIGN KEY(session_id) REFERENCES chat_sessions(id))";
        db.execSQL(CREATE_MESSAGES_TABLE);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS chat_messages");
        db.execSQL("DROP TABLE IF EXISTS chat_sessions");
        onCreate(db);
    }

    public long startNewSession(String title) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put("title", title);
        currentSessionId = db.insert("chat_sessions", null, values);
        return currentSessionId;
    }

    public void addMessage(String message, String sender) {
        if (currentSessionId == -1) {
            startNewSession("Chat with Assistant");
        }
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put("session_id", currentSessionId);
        values.put("message", message);
        values.put("sender", sender);
        db.insert("chat_messages", null, values);
    }

    public Cursor getAllSessions() {
        SQLiteDatabase db = this.getReadableDatabase();
        return db.rawQuery("SELECT * FROM chat_sessions ORDER BY timestamp DESC", null);
    }

    public Cursor getMessagesForSession(long sessionId) {
        SQLiteDatabase db = this.getReadableDatabase();
        return db.rawQuery("SELECT message, sender, timestamp FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC",
                new String[]{String.valueOf(sessionId)});
    }

    public void addMessageToChat(String firstMessage, String message, String sentBy, String timestamp) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put("section_id", currentSessionId); // Store the section ID
        values.put("first_message", firstMessage);
        values.put("message", message);
        values.put("sent_by", sentBy);
        values.put("timestamp", timestamp);

        db.insert("chat_table", null, values);
        db.close();
    }


    public void endSession() {
        currentSessionId = -1;
    }
}
