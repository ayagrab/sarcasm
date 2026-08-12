export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type ModelStatus =
  | "AVAILABLE"
  | "NOT_TRAINED_YET"
  | "NOT_FROZEN_YET"
  | "UNAVAILABLE";

export interface PredictResponse {
  label: "sarcastic" | "not_sarcastic";
  confidence: number | null;
  model: string;
  runtime_seconds: number;
}

export interface MethodPrediction {
  method: string;
  display_name: string;
  status: ModelStatus;
  label: "sarcastic" | "not_sarcastic" | null;
  confidence: number | null;
  runtime_seconds: number | null;
  error: string | null;
}

export interface CompareResponse {
  text: string;
  predictions: MethodPrediction[];
  agreement: boolean | null;
}

export interface MethodInfo {
  method: string;
  display_name: string;
  status: ModelStatus;
  description: string;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail ? JSON.stringify(data.detail) : detail;
    } catch {
      // response body wasn't JSON -- keep the statusText fallback
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export function predict(text: string): Promise<PredictResponse> {
  return postJson<PredictResponse>("/predict", { text });
}

export function compare(text: string): Promise<CompareResponse> {
  return postJson<CompareResponse>("/compare", { text });
}

export async function listMethods(): Promise<MethodInfo[]> {
  const response = await fetch(`${API_BASE_URL}/methods`);
  if (!response.ok) {
    throw new ApiError(response.statusText, response.status);
  }
  return response.json() as Promise<MethodInfo[]>;
}
