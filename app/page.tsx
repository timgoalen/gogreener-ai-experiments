"use client";

import { useState } from "react";

import Markdown from "react-markdown";
import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} from "@google/generative-ai";

export default function Home() {
  const [response, setResponse] = useState("");
  const [userTextInputValue, setUserTextInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY as string;
  const genAI = new GoogleGenerativeAI(apiKey);

  const model = genAI.getGenerativeModel({
    model: "gemini-1.5-pro-latest",
    // Try other system instructions and compare
    // (eg. "You are a helpful assistant with the GoGreener web framework. Only answer questions about GoGreener and web developmewnt in general")
    systemInstruction: "You are an expert in the Golang programming language",
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

  async function run() {
    setIsLoading(true);

    const chatSession = model.startChat({
      generationConfig,
      safetySettings,
      history: [],
    });

    const result = await chatSession.sendMessage(userTextInputValue);
    // const result = await chatSession.sendMessage("INSERT_INPUT_HERE");
    setResponse(result.response.text());
    // console.log(result.response.text());
    // console.log(response);
    // console.log(result);
    setIsLoading(false);
    setUserTextInputValue("");
  }

  function handleFormChange(event) {
    setUserTextInputValue(event.target.value);
  }

  return (
    <main>
      <h1>GoGreener</h1>
      <h2>Ai Helper</h2>

      {isLoading && <div>Sending request...</div>}

      {response && (
        <section className="response">
          <Markdown children={response} />
          {/* {console.log(response)} */}
        </section>
      )}

      <div className="input">
        <input value={userTextInputValue} onChange={handleFormChange} />
        <button onClick={() => run()}>Send</button>
      </div>
    </main>
  );
}
