"use client";

import { useState } from "react";
import { Roboto_Mono } from "next/font/google";

import Markdown from "react-markdown";
import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} from "@google/generative-ai";
import { ChevronUp } from "lucide-react";

const robotoMono = Roboto_Mono({ subsets: ["latin"] });

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
    setResponse(result.response.text());
    setIsLoading(false);
    setUserTextInputValue("");
  }

  function handleFormChange(event) {
    setUserTextInputValue(event.target.value);
  }

  return (
    <>
      <header>
        <h1 className="title">GoGreener</h1>
        <div>
          <h2 className="subtitle">AI Helper</h2>
        </div>
      </header>

      <main>
        {isLoading && <div>Sending request...</div>}

        <section className="response">
          {response && <Markdown children={response} />}
        </section>

        <div className="input-container">
          <input
            className={`${robotoMono.className} input`}
            value={userTextInputValue}
            onChange={handleFormChange}
            placeholder="Enter your question"
          />

          <button className="submit-btn" onClick={() => run()}>
            {/* <ChevronUp className="btn-icon" color="white" size={24} /> */}
            <ChevronUp className="btn-icon" size={24} />
          </button>

          <div className="small-glow"></div>
          <div className="big-glow"></div>
        </div>
      </main>
    </>
  );
}
