package com.example.navigationdrawer;

import android.os.Bundle;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import java.util.ArrayList;
import java.util.List;

public class HelpFragment extends Fragment {

    private RecyclerView rvFAQ;
    private FAQAdapter faqAdapter;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_help, container, false);
        rvFAQ = view.findViewById(R.id.rvFAQ);

        // Setting up RecyclerView
        rvFAQ.setLayoutManager(new LinearLayoutManager(getContext()));
        faqAdapter = new FAQAdapter(getFAQs());
        rvFAQ.setAdapter(faqAdapter);

        return view;
    }

    // Method to get FAQ data
    private List<FAQ> getFAQs() {
        List<FAQ> faqs = new ArrayList<>();
         faqs.add(new FAQ("How do I start a new chat?",
                "You can start a new chat by selecting 'New Chat' in the navigation bar."));
        faqs.add(new FAQ("How accurate are the chatbot's responses?",
                "The chatbot is designed to give accurate and relevant responses based on available university information, but please reach out to a university representative for complex or detailed inquiries."));
        faqs.add(new FAQ("Is my chat history saved?",
                "Yes, chat history is saved locally and can be accessed via the \"Chat History\" option. It helps you review previous interactions for reference."));
        faqs.add(new FAQ("What is this chatbot application for?",
                "This chatbot is designed to provide quick answers and assist with common questions."));
        faqs.add(new FAQ("How do I report a problem with the chatbot?",
                "ou can report issues via the \"Feedback\" option in the navigation bar. Provide details of the problem, and the support team will address it."));
        faqs.add(new FAQ("What is this chatbot application for?",
                "This chatbot is designed to provide quick answers and assist with common questions."));
        faqs.add(new FAQ("Is the chatbot’s data secure?",
                " Yes, data privacy is a priority. We follow industry standards to keep your information secure."));
        faqs.add(new FAQ("How do I update the chatbot application?",
                "Updates are available via the app store. You’ll receive a notification whenever a new version is ready."));
        faqs.add(new FAQ("Can I access the chatbot on multiple devices?",
                "Currently, the chatbot is available only on this device. Any chat history or settings will not sync across different devices."));
        // Add more FAQs here
        return faqs;
    }

    // Adapter class for FAQ
    private static class FAQAdapter extends RecyclerView.Adapter<FAQAdapter.FAQViewHolder> {
        private final List<FAQ> faqList;

        FAQAdapter(List<FAQ> faqList) {
            this.faqList = faqList;
        }

        @NonNull
        @Override
        public FAQViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.faq_item, parent, false);
            return new FAQViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull FAQViewHolder holder, int position) {
            FAQ faq = faqList.get(position);
            holder.tvQuestion.setText(faq.getQuestion());
            holder.tvAnswer.setText(faq.getAnswer());
        }

        @Override
        public int getItemCount() {
            return faqList.size();
        }

        // ViewHolder for FAQ items
        // ViewHolder for FAQ items
        static class FAQViewHolder extends RecyclerView.ViewHolder {
            TextView tvQuestion;
            TextView tvAnswer;

            FAQViewHolder(View itemView) {
                super(itemView);
                tvQuestion = itemView.findViewById(R.id.tvQuestion);
                tvAnswer = itemView.findViewById(R.id.tvAnswer);
            }
        }

    }

    // FAQ Model Class
    private static class FAQ {
        private final String question;
        private final String answer;

        FAQ(String question, String answer) {
            this.question = question;
            this.answer = answer;
        }

        public String getQuestion() {
            return question;
        }

        public String getAnswer() {
            return answer;
        }
    }
}
