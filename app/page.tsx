"use client";

import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} from "@google/generative-ai";

export default function Home() {
  // const apiKey = process.env.GEMINI_API_KEY;
  const apiKey = "API_KEY_GOES_HERE";
  const genAI = new GoogleGenerativeAI(apiKey);

  const model = genAI.getGenerativeModel({
    model: "gemini-1.5-pro-latest",
    // Try other system instructions and compare 
    // (eg. "You are a helpful assistant with the GoGreener web framework. Only answer questions about GoGreener and web developmewnt in general")
    systemInstruction: "You are an expert in the GoGreener web framework, built with Go. The github repo is: https://github.com/thejimmyg/greener",
  });

  const generationConfig = {
    temperature: 1,
    topP: 0.95,
    topK: 64,
    maxOutputTokens: 8192,
    responseMimeType: "text/plain",
  };

  const safetySettings = [
    {
      category: HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
  ];

  let response = "";

  async function run() {
    const chatSession = model.startChat({
      generationConfig,
      safetySettings,
      history: [], 
    });

    const result = await chatSession.sendMessage("how do i link a css file");
    // const result = await chatSession.sendMessage("INSERT_INPUT_HERE");
    response = result.response.text();
    // console.log(result.response.text());
    console.log(response);
  }

  return (
    <main>
      <h1>GoGreener AI Helper</h1>

      <input></input>
      <button onClick={() => run()}>Ask the AI</button>
      {/* {result && (

      )}  */}
    </main>
  );
}
