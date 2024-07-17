import {GoogleGenerativeAI} from "@google/generative-ai";
export const configureAI = () => {
    const genAI = new GoogleGenerativeAI(process.env.API_KEY);

    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});
    
  }; 


// Access your API key as an environment variable (see "Set up your API key" above)
