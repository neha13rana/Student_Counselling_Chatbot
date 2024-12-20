package com.example.navigationdrawer;


import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.IOException;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;




import java.io.InputStream;


import java.io.BufferedReader;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;



import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.EditText;
import android.widget.Toast;

import org.apache.poi.hwpf.HWPFDocument;
import org.apache.poi.hwpf.extractor.WordExtractor;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;
import org.apache.poi.xwpf.usermodel.XWPFDocument;



import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.MediaType;
import okhttp3.Response;

import org.json.JSONArray;
import org.json.JSONObject;


public class HomeFragment extends Fragment {
    private static final int PICK_FILE_REQUEST_CODE = 100; // Add this line

    RecyclerView recyclerView;
    TextView welcomeTextView;
    EditText messageEditText;
    ImageButton sendButton,uploadButton;
    List<ChatMessage> messageList;
    MessageAdapter messageAdapter;
    ChatDatabaseHelper db;



    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment

        setRetainInstance(true);  // Retain fragment instance across fragment transactions

        View view = inflater.inflate(R.layout.fragment_home, container, false);

        db = new ChatDatabaseHelper(getContext());

        recyclerView = view.findViewById(R.id.recycler_view);
        welcomeTextView = view.findViewById(R.id.welcome_text);
        messageEditText = view.findViewById(R.id.message_edit_text);
        sendButton = view.findViewById(R.id.send_btn);
        uploadButton = view.findViewById(R.id.upload_btn);

        messageList = new ArrayList<>();

        messageAdapter = new MessageAdapter(messageList);
        recyclerView.setAdapter(messageAdapter);
        LinearLayoutManager llm = new LinearLayoutManager(getContext());
        llm.setStackFromEnd(true);
        recyclerView.setLayoutManager(llm);

        sendButton.setOnClickListener((v) -> {
            String question = messageEditText.getText().toString().trim();
            addToChat(question, ChatMessage.SENT_BY_ME);
            db.addMessage(question, ChatMessage.SENT_BY_ME); // Store the question in the database
            callAPI(question);
            messageEditText.setText("");
            welcomeTextView.setVisibility(View.GONE);

        });

        // Return the inflated view
        uploadButton.setOnClickListener(v -> openFileChooser());
        return view;
    }



    private void openFileChooser() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        startActivityForResult(intent, PICK_FILE_REQUEST_CODE);
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_FILE_REQUEST_CODE && resultCode == getActivity().RESULT_OK && data != null) {
            Uri fileUri = data.getData();
            String fileName = getFileName(fileUri);
            File file = new File(getContext().getCacheDir(), fileName);

            try (InputStream inputStream = getContext().getContentResolver().openInputStream(fileUri);
                 FileOutputStream outputStream = new FileOutputStream(file)) {
                byte[] buffer = new byte[1024];
                int length;
                while ((length = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, length);
                }

                processFileAndSendText(file,fileUri); // For other file types like DOC, DOCX, etc.

            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(getContext(), "File upload failed", Toast.LENGTH_SHORT).show();
            }
        }
    }
    private String getFileName(Uri uri) {
        String fileName = "";
        try (Cursor cursor = getContext().getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int columnIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if(columnIndex>=0) {
                    fileName = cursor.getString(columnIndex);
                }
            }
        }
        return fileName;
    }


    private void processFileAndSendText(File file,Uri uri) {
        String filePath = file.getPath().toLowerCase();
        db.addMessage(getFileName(uri), ChatMessage.SENT_BY_ME);

        if (filePath.endsWith(".doc") || filePath.endsWith(".docx")) {
            processWordFile(file);
        } else {
            processTextFile(file);
        }
    }


    private void processWordFile(File file) {
        String extractedText = "";
        String filePath = file.getPath().toLowerCase();

        try (FileInputStream fis = new FileInputStream(file)) {
            if (filePath.endsWith(".doc")) {
                // For .doc files
                HWPFDocument doc = new HWPFDocument(fis);
                WordExtractor extractor = new WordExtractor(doc);
                extractedText = extractor.getText().trim();
                extractor.close();
                doc.close();
            } else if (filePath.endsWith(".docx")) {
                // For .docx files
                XWPFDocument docx = new XWPFDocument(fis);
                XWPFWordExtractor extractor = new XWPFWordExtractor(docx);
                extractedText = extractor.getText().trim();
                extractor.close();
                docx.close();
            }

            // Send extracted text to Gemini API
            if (!extractedText.isEmpty()) {
                messageList.add(new ChatMessage("File Uploaded Successfully.", ChatMessage.SENT_BY_ME));
                callAPI(extractedText);  // Use your existing callAPI method
            } else {
                addToChat("Word file is empty or unreadable.", ChatMessage.SENT_BY_BOT);
            }

        } catch (Exception e) {
            e.printStackTrace();
            addToChat("Error processing Word file: " + e.getMessage(), ChatMessage.SENT_BY_BOT);
        }
    }

    private void processTextFile(File file) {
        try {
            StringBuilder text = new StringBuilder();
            BufferedReader br = new BufferedReader(new FileReader(file));
            String line;

            while ((line = br.readLine()) != null) {
                text.append(line).append("\n"); // Append each line with a newline character
            }
            br.close();

            String extractedText = text.toString().trim();

            // Send extracted text to Gemini API or display it
            if (!extractedText.isEmpty()) {
                addToChat("File Uploaded Successfully.", ChatMessage.SENT_BY_ME);
                callAPI(extractedText);  // Use your existing callAPI method
            } else {
                addToChat("Text file is empty or unreadable.", ChatMessage.SENT_BY_BOT);
            }

        } catch (Exception e) {
            e.printStackTrace();
            addToChat("Error processing text file: " + e.getMessage(), ChatMessage.SENT_BY_BOT);
        }
    }




    void addToChat(String message, String sentBy) {
        // Use getActivity() to access the parent activity for UI operations
        getActivity().runOnUiThread(() -> {
            messageList.add(new ChatMessage(message, sentBy));
            messageAdapter.notifyDataSetChanged();
            recyclerView.smoothScrollToPosition(messageAdapter.getItemCount());
        });
    }
    void addResponse(String response) {
        messageList.remove(messageList.size() - 1); // Remove "Typing..." message
        addToChat(response, ChatMessage.SENT_BY_BOT);
        db.addMessage(response, ChatMessage.SENT_BY_BOT);
    }

    void callAPI(String question) {
        // Add "Typing..." message while waiting for the response
        messageList.add(new ChatMessage("Typing...", ChatMessage.SENT_BY_BOT));

        // Send the user's input to the Hugging Face API
        com.example.navigationdrawer.HuggingFaceAPI huggingFaceAPI = new com.example.navigationdrawer.HuggingFaceAPI();

        // Use the Hugging Face model to get the prediction
        huggingFaceAPI.getPrediction(question, new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // Handle failure, log the error and update the chat
                getActivity().runOnUiThread(() -> {
                    messageList.remove(messageList.size() - 1); // Remove "Typing..." message
                    addToChat("Error: " + e.getMessage(), ChatMessage.SENT_BY_BOT);
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    // Get the raw response string
                    String modelResponse = response.body().string();

                    // Log the raw response for debugging purposes
                    Log.d("Response", modelResponse);

                    try {
                        // Clean up the response to extract the JSON part
                        String cleanedResponse = modelResponse.substring(modelResponse.indexOf("data:") + 5).trim();
                        cleanedResponse = cleanedResponse.trim().replaceAll("^[\\n\\r]*", "").replaceAll("[\\n\\r]*$", "");

                        // The cleaned response should now be a JSON array
                        JSONArray dataArray = new JSONArray(cleanedResponse);

                        // Get the first element of the "data" array (it's the string you're interested in)
                        String responseData = dataArray.getString(0);

                        // Update the chat with the extracted response
                        getActivity().runOnUiThread(() -> {
                            messageList.remove(messageList.size() - 1); // Remove "Typing..." message
                            addToChat(responseData, ChatMessage.SENT_BY_BOT); // Add response to the chat
                            addResponse(responseData); // Save or log the response (if necessary)
                        });

                    } catch (Exception e) {
                        // Log any errors that occur during parsing
                        Log.e("Parsing Error", "Failed to parse response", e);
                        getActivity().runOnUiThread(() -> {
                            messageList.remove(messageList.size() - 1); // Remove "Typing..." message
                            addToChat("Failed to parse response", ChatMessage.SENT_BY_BOT);
                        });
                    }
                } else {
                    // Handle unsuccessful response
                    getActivity().runOnUiThread(() -> {
                        messageList.remove(messageList.size() - 1); // Remove "Typing..." message
                        addToChat("Failed to get response", ChatMessage.SENT_BY_BOT);
                    });
                }
            }
        });
    }


}
