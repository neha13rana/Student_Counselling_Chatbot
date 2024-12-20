package com.example.navigationdrawer;

import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;

public class HuggingFaceAPI {

    private static final String BASE_URL = "https://ashu-03-model.hf.space/gradio_api/call/predict";
    private static final String EVENT_URL = "https://ashu-03-model.hf.space/gradio_api/call/predict/";

    private OkHttpClient client = new OkHttpClient();

    // Function to send a POST request
    public void getPrediction(String userInput, Callback callback) {
        // Create JSON object for the POST request body
        JSONObject json = new JSONObject();
        JSONArray dataArray = new JSONArray();
        dataArray.put(userInput);
        try {
            json.put("data", dataArray);

            RequestBody body = RequestBody.create(
                    json.toString(), MediaType.parse("application/json"));
            Request request = new Request.Builder()
                    .url(BASE_URL)
                    .post(body)
                    .build();

            // Send POST request
            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    if (response.isSuccessful()) {
                        // Extract EVENT_ID from the response
                        String responseString = response.body().string();
                        String eventId = parseEventId(responseString);

                        // Fetch results using the GET request
                        getResults(eventId, callback);
                    } else {
                        callback.onFailure(call, new IOException("Unexpected code " + response));
                    }
                }

                @Override
                public void onFailure(Call call, IOException e) {
                    callback.onFailure(call, e);
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private void getResults(String eventId, Callback callback) {
        Request request = new Request.Builder()
                .url(EVENT_URL + eventId)
                .build();

        client.newCall(request).enqueue(callback);
    }

    // Helper function to parse the EVENT_ID from the POST response
    private String parseEventId(String responseString) {
        // Assuming the response is in JSON format with the event_id
        try {
            JSONObject responseJson = new JSONObject(responseString);
            return responseJson.getString("event_id");
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}