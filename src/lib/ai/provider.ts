import { AIProviderError, RateLimitError } from '../errors';
import { rateLimiter } from '../rate-limiter';

export type AIMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

export type AIOptions = {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
};

export type AIResponse = {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
};

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';
const DEFAULT_MODEL = process.env.AI_MODEL || 'llama3.2';

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries = 3,
  attempt = 1
): Promise<Response> {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      const errorBody = await response.text().catch(() => 'Unknown error');
      throw new Error(`HTTP ${response.status}: ${errorBody}`);
    }
    return response;
  } catch (error) {
    if (attempt >= maxRetries) throw error;
    const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
    await sleep(delay);
    return fetchWithRetry(url, options, maxRetries, attempt + 1);
  }
}

async function callOllama(
  messages: AIMessage[],
  options: AIOptions
): Promise<{ content: string; usage: AIResponse['usage'] }> {
  const model = options.model || DEFAULT_MODEL;
  const body = {
    model,
    messages,
    temperature: options.temperature ?? 0.7,
    max_tokens: options.maxTokens ?? 4096,
    stream: false,
  };

  const response = await fetchWithRetry(`${OLLAMA_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  const choice = data.choices?.[0];
  if (!choice?.message?.content) {
    throw new AIProviderError('ollama', 'No content in response', data);
  }

  return {
    content: choice.message.content,
    usage: {
      promptTokens: data.usage?.prompt_tokens ?? estimateTokens(messages.map((m) => m.content).join('')),
      completionTokens: data.usage?.completion_tokens ?? estimateTokens(choice.message.content),
      totalTokens: data.usage?.total_tokens ?? 0,
    },
  };
}

async function callOllamaStream(
  messages: AIMessage[],
  options: AIOptions
): Promise<ReadableStream<Uint8Array>> {
  const model = options.model || DEFAULT_MODEL;
  const body = {
    model,
    messages,
    temperature: options.temperature ?? 0.7,
    max_tokens: options.maxTokens ?? 4096,
    stream: true,
  };

  const response = await fetchWithRetry(`${OLLAMA_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.body) {
    throw new AIProviderError('ollama', 'No response body');
  }

  return new ReadableStream({
    async start(controller) {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n').filter((l) => l.startsWith('data: '));
          for (const line of lines) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ content })}\n\n`));
              }
            } catch {
              // skip invalid JSON lines
            }
          }
        }
      } finally {
        controller.close();
      }
    },
  });
}

export async function chat(messages: AIMessage[], options: AIOptions = {}): Promise<AIResponse> {
  await rateLimiter.enforce('ai:ollama', 20, 60_000);

  try {
    return await callOllama(messages, options);
  } catch (error) {
    if (error instanceof RateLimitError || error instanceof AIProviderError) throw error;
    throw new AIProviderError('ollama', (error as Error).message || 'Request failed', error);
  }
}

export async function generate(prompt: string, options: AIOptions = {}): Promise<AIResponse> {
  return chat([{ role: 'user', content: prompt }], options);
}

export async function chatStream(
  messages: AIMessage[],
  options: AIOptions = {}
): Promise<ReadableStream<Uint8Array>> {
  await rateLimiter.enforce('ai:ollama', 20, 60_000);

  try {
    return await callOllamaStream(messages, options);
  } catch (error) {
    if (error instanceof RateLimitError || error instanceof AIProviderError) throw error;
    throw new AIProviderError('ollama', (error as Error).message || 'Stream failed', error);
  }
}

export function countTokens(text: string): number {
  return estimateTokens(text);
}
