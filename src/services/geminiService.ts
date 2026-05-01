import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export interface AnalysisResponse {
  classification: 'REAL' | 'FAKE' | 'SUSPICIOUS';
  confidence: number;
  reasoning: string[];
  keyFlags: string[];
  summary: string;
}

export async function analyzeNews(text: string): Promise<AnalysisResponse> {
  const prompt = `
    Analyze the following news text for veracity. 
    Classify it as REAL, FAKE, or SUSPICIOUS.
    Provide a confidence score between 0 and 1.
    List key flags (reasons for suspicion or markers of authenticity).
    Provide a brief summary of the reasoning.

    Output in JSON format:
    {
      "classification": "REAL" | "FAKE" | "SUSPICIOUS",
      "confidence": number,
      "reasoning": string[],
      "keyFlags": string[],
      "summary": string
    }

    News Text:
    "${text}"
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    
    return {
      classification: parsed.classification || 'SUSPICIOUS',
      confidence: parsed.confidence || 0.5,
      reasoning: parsed.reasoning || [],
      keyFlags: parsed.keyFlags || [],
      summary: parsed.summary || 'Failed to generate a detailed summary.'
    };
  } catch (error) {
    console.error("AI Analysis Error:", error);
    throw new Error("Failed to reach AI Analysis Engine.");
  }
}
