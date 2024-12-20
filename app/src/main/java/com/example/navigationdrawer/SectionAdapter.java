package com.example.navigationdrawer;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class SectionAdapter extends RecyclerView.Adapter<SectionAdapter.ViewHolder> {
    private List<ChatSection> sectionList;

    public SectionAdapter(List<ChatSection> sectionList) {
        this.sectionList = sectionList;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_chat_section, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(final ViewHolder holder, int position) {
        ChatSection section = sectionList.get(position);
        holder.titleTextView.setText(section.getTitle());
        holder.timestampTextView.setText(section.getTimestamp());

        // Toggle visibility of messages list
        holder.messageListTextView.setVisibility(section.isExpanded() ? View.VISIBLE : View.GONE);

        // Click listener for expanding/collapsing the section
        holder.titleTextView.setOnClickListener(v -> {
            section.setExpanded(!section.isExpanded());
            notifyItemChanged(position);
        });

        if (section.isExpanded()) {
            // Display all messages with formatting
            StringBuilder messages = new StringBuilder();
            for (String message : section.getMessages()) {
                messages.append("- ").append(message).append("\n");
            }
            holder.messageListTextView.setText(messages.toString().trim());  // Trim trailing newline
        }
    }

    @Override
    public int getItemCount() {
        return sectionList.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        TextView titleTextView, timestampTextView, messageListTextView;

        public ViewHolder(View itemView) {
            super(itemView);
            titleTextView = itemView.findViewById(R.id.section_title);
            timestampTextView = itemView.findViewById(R.id.section_timestamp);
            messageListTextView = itemView.findViewById(R.id.section_messages);
        }
    }
}
