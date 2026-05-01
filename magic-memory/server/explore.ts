const envFile = Bun.file(import.meta.dir + "/../../.env");
if (await envFile.exists()) {
  const text = await envFile.text();
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIndex = trimmed.indexOf("=");
    if (eqIndex === -1) continue;
    const key = trimmed.slice(0, eqIndex).trim();
    const value = trimmed.slice(eqIndex + 1).trim();
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

const PORT = 4321;

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || "";
const DEEPSEEK_BASE_URL = process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1";

interface ExploreRequest {
  sourceConcept: {
    id: string;
    title: string;
    problem?: string;
  };
  userQuestion: string;
  relationType: "leads_to" | "depends_on" | "related";
}

interface ExploreResponse {
  title: string;
  problem: string;
  gap_anticipate: string;
  content: string;
  relationType: "leads_to" | "depends_on" | "related";
}

const SYSTEM_PROMPT = `你是一个知识图谱概念生成助手。你的任务是基于用户提供的源概念和探索问题，生成一个新的概念节点。

输出必须是一个 JSON 对象，包含以下字段：
- title: 概念名称（简短、精确，10个字以内）
- problem: 这个概念解决的核心问题（一句话）
- gap_anticipate: 学习这个概念时可能产生的疑问（一句话，30字以内）
- content: 概念的详细内容，用 Markdown 格式（至少 200 字，包含概述、核心原理、关键要点）

只输出 JSON，不要包含其他文字。`;

async function callLLM(userPrompt: string): Promise<string> {
  const resp = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${DEEPSEEK_API_KEY}`,
    },
    body: JSON.stringify({
      model: "deepseek-chat",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.7,
      max_tokens: 2000,
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text().catch(() => "unknown error");
    throw new Error(`LLM API error (${resp.status}): ${errText}`);
  }

  const data = await resp.json();
  const text = data.choices?.[0]?.message?.content;
  if (!text) {
    throw new Error(`No content in LLM response: ${JSON.stringify(data)}`);
  }
  return text;
}

function parseJSONResponse(text: string): ExploreResponse {
  try {
    return JSON.parse(text);
  } catch {
    const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[1].trim());
    }
    throw new Error(`Failed to parse LLM response as JSON: ${text.slice(0, 200)}`);
  }
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    if (req.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    if (req.method !== "POST" || new URL(req.url).pathname !== "/api/explore") {
      return new Response("Not Found", { status: 404 });
    }

    try {
      const body: ExploreRequest = await req.json();
      const userPrompt = `源概念: ${body.sourceConcept.title}
${body.sourceConcept.problem ? `源概念问题: ${body.sourceConcept.problem}` : ""}
探索问题: ${body.userQuestion}

请基于源概念「${body.sourceConcept.title}」和探索问题「${body.userQuestion}」，生成一个新的概念。输出 JSON。`;

      const rawResponse = await callLLM(userPrompt);
      const result = parseJSONResponse(rawResponse);

      return new Response(JSON.stringify(result), {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (error: any) {
      console.error("Explore error:", error);
      return new Response(
        JSON.stringify({ error: error.message || "Internal server error" }),
        {
          status: 500,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
          },
        }
      );
    }
  },
});

console.log(`🚀 Explore server running on http://localhost:${PORT}`);
console.log(`   (DeepSeek API: ${DEEPSEEK_BASE_URL})`);
