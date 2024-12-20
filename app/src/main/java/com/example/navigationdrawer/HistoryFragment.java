package com.example.navigationdrawer;

import android.database.Cursor;
import android.os.Bundle;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import java.util.ArrayList;
import java.util.List;

public class HistoryFragment extends Fragment {
    RecyclerView recyclerView;
    SectionAdapter sectionAdapter;
    List<ChatSection> sectionList;
    ChatDatabaseHelper db;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_history, container, false);
        recyclerView = view.findViewById(R.id.recycler_view_history);
        sectionList = new ArrayList<>();
        sectionAdapter = new SectionAdapter(sectionList);
        recyclerView.setAdapter(sectionAdapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));

        db = new ChatDatabaseHelper(getContext());
        loadChatSessions();
        return view;
    }

    private void loadChatSessions() {
        Cursor sessionCursor = db.getAllSessions();
        if (sessionCursor != null && sessionCursor.moveToFirst()) {
            do {
                long sessionId = sessionCursor.getLong(sessionCursor.getColumnIndex("id"));
                String title = sessionCursor.getString(sessionCursor.getColumnIndex("title"));
                String timestamp = sessionCursor.getString(sessionCursor.getColumnIndex("timestamp"));

                ChatSection section = new ChatSection(sessionId, title, timestamp);

                // Load messages for each session
                Cursor messagesCursor = db.getMessagesForSession(sessionId);
                if (messagesCursor != null && messagesCursor.moveToFirst()) {
                    do {
                        String message = messagesCursor.getString(messagesCursor.getColumnIndex("message"));
                        String sender = messagesCursor.getString(messagesCursor.getColumnIndex("sender"));
                        section.addMessage(message, sender);
                    } while (messagesCursor.moveToNext());
                    messagesCursor.close();
                }

                sectionList.add(section);
            } while (sessionCursor.moveToNext());
            sessionCursor.close();
        }
        sectionAdapter.notifyDataSetChanged();
    }
}
