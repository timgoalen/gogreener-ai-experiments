"use client";

import { useState, useRef, ChangeEvent, KeyboardEvent } from "react";
import { Roboto_Mono } from "next/font/google";

import Markdown from "react-markdown";
import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} from "@google/generative-ai";
import { CircleArrowUp } from "lucide-react";

const robotoMono = Roboto_Mono({ subsets: ["latin"] });

export default function Home() {
  const [response, setResponse] = useState("");
  const [userTextInputValue, setUserTextInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);

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

  // Store the form text in state
  function handleFormChange(event: ChangeEvent<HTMLTextAreaElement>) {
    setUserTextInputValue(event.target.value);
  }

  // Auto resize the textarea based on input.
  function handleInput(event: ChangeEvent<HTMLTextAreaElement>) {
    if (ref.current) {
      ref.current.style.height = "auto";
      ref.current.style.height = `${event.target.scrollHeight}px`;
    }
  }

  // 'Shift-Return' for new line, 'Return' to send form.
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.shiftKey) {
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      console.log("Submitting form:", userTextInputValue);
      run();
      if (ref.current) {
        // Reset textarea height
        ref.current.style.height = "auto";
      }
      // Reset textarea content
      setUserTextInputValue("");
    }
  }

  // Function to submit form with button click (change for server action)
  function submitForm(event: ChangeEvent<HTMLFormElement>) {
    event.preventDefault();
    console.log("Submitting form:", userTextInputValue);
    run();
    if (ref.current) {
      // Reset textarea height
      ref.current.style.height = "auto";
    }
    // Reset textarea content
    setUserTextInputValue("");
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

        <div className="page-container">
          <section className="response-container">
            <div className="response-content">
              {response && <Markdown>{response}</Markdown>}
            </div>
          </section>

          <section className="input-container">
            <form action="" onSubmit={submitForm} className="form">
              <textarea
                ref={ref}
                className={`${robotoMono.className}`}
                name="user-prompt"
                id="text-area"
                rows={1}
                placeholder="Enter your question"
                tabIndex={0}
                value={userTextInputValue}
                onInput={handleInput}
                onChange={handleFormChange}
                onKeyDown={handleKeyDown}
              ></textarea>
              <div className="btn-container">
                <div className="small-glow"></div>
                <button type="submit" className="submit-btn">
                  <CircleArrowUp className="btn-icon" size={24} />
                </button>
              </div>
              <div className="big-glow"></div>
            </form>
          </section>
        </div>
      </main>
    </>
  );
}
