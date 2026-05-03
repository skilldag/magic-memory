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

import {
  listProjects,
  registerProject,
  removeProject,
  saveGraph,
  loadGraph,
} from "./graphBuilder";

import type { ProjectMeta, Concept, ConceptEdge } from "./graphBuilder";

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || "";
const DEEPSEEK_BASE_URL = process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1";

const EXPLORE_SYSTEM_PROMPT = `你是一个知识图谱概念生成助手。你的任务是基于用户提供的源概念和探索问题，生成一个新的概念节点。
输出必须是一个 JSON 对象，包含以下字段：
- title: 概念名称（简短、精确，10个字以内）
- problem: 这个概念解决的核心问题（一句话）
- gap_anticipate: 学习这个概念时可能产生的疑问（一句话，30字以内）
- content: 概念的详细内容，用 Markdown 格式（至少 200 字，包含概述、核心原理、关键要点）
只输出 JSON，不要包含其他文字。`;

async function handleExplore(req: Request): Promise<Response> {
  try {
    const body = await req.json() as any;
    const userPrompt = `源概念: ${body.sourceConcept.title}
${body.sourceConcept.problem ? `源概念问题: ${body.sourceConcept.problem}` : ""}
探索问题: ${body.userQuestion}
请基于源概念「${body.sourceConcept.title}」和探索问题「${body.userQuestion}」，生成一个新的概念。输出 JSON。`;

    const resp = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: "deepseek-chat",
        messages: [
          { role: "system", content: EXPLORE_SYSTEM_PROMPT },
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

    const data = await resp.json() as any;
    const text = data.choices?.[0]?.message?.content;
    if (!text) throw new Error("No content in LLM response");

    let result: any;
    try {
      result = JSON.parse(text);
    } catch {
      const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) result = JSON.parse(jsonMatch[1].trim());
      else throw new Error("Failed to parse LLM response as JSON");
    }

    return new Response(JSON.stringify(result), {
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  } catch (error: any) {
    console.error("Explore error:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  }
}

async function handleListProjects(): Promise<Response> {
  try {
    const projects = await listProjects();
    return new Response(JSON.stringify({ projects }), {
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  }
}

async function handleCreateProject(req: Request): Promise<Response> {
  try {
    const body = await req.json() as {
      id?: string;
      name: string;
      sourceDir: string;
      concepts?: Concept[];
      edges?: ConceptEdge[];
    };

    if (!body.name || body.sourceDir === undefined) {
      return new Response(JSON.stringify({ error: "name and sourceDir required" }), {
        status: 400,
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
      });
    }

    const projectId = body.id || `proj_${Date.now()}`;
    const meta: ProjectMeta = {
      id: projectId,
      name: body.name,
      sourceDir: body.sourceDir,
      createdAt: new Date().toISOString(),
      conceptCount: body.concepts?.length || 0,
      edgeCount: body.edges?.length || 0,
    };

    await registerProject(meta);

    if (body.concepts && body.edges) {
      await saveGraph(projectId, body.concepts, body.edges);
    }

    return new Response(JSON.stringify(meta), {
      status: 201,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  }
}

async function handleDeleteProject(projectId: string): Promise<Response> {
  try {
    await removeProject(projectId);
    return new Response(JSON.stringify({ success: true }), {
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  }
}

async function handleGetGraph(projectId: string): Promise<Response> {
  try {
    const graph = await loadGraph(projectId);
    if (!graph) {
      return new Response(JSON.stringify({ error: "Graph not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
      });
    }
    return new Response(JSON.stringify(graph), {
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  }
}

async function handleUpdateGraph(projectId: string, req: Request): Promise<Response> {
  try {
    const body = await req.json() as { concepts: Concept[]; edges: ConceptEdge[] };
    if (!body.concepts || !body.edges) {
      return new Response(JSON.stringify({ error: "concepts and edges required" }), {
        status: 400,
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
      });
    }
    await saveGraph(projectId, body.concepts, body.edges);
    return new Response(JSON.stringify({
      success: true,
      conceptCount: body.concepts.length,
      edgeCount: body.edges.length,
    }), {
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
    });
  }
}

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function corsResponse(status: number, body?: any): Response {
  return new Response(body ? JSON.stringify(body) : null, {
    status,
    headers: { ...CORS_HEADERS, "Content-Type": "application/json" },
  });
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const method = req.method;

    if (method === "OPTIONS") {
      return corsResponse(200);
    }

    if (url.pathname === "/api/explore" && method === "POST") {
      return handleExplore(req);
    }
    if (url.pathname === "/api/projects" && method === "GET") {
      return handleListProjects();
    }
    if (url.pathname === "/api/projects" && method === "POST") {
      return handleCreateProject(req);
    }
    const graphMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)\/graph$/);
    if (graphMatch && method === "GET") {
      return handleGetGraph(graphMatch[1]);
    }
    if (graphMatch && method === "PUT") {
      return handleUpdateGraph(graphMatch[1], req);
    }
    const projectMatch = url.pathname.match(/^\/api\/projects\/([^\/]+)$/);
    if (projectMatch && method === "DELETE") {
      return handleDeleteProject(projectMatch[1]);
    }

    return corsResponse(404, { error: "Not Found" });
  },
});

console.log(`🚀 Global Service running on http://localhost:${PORT}`);
console.log(`   Explore API: POST /api/explore`);
console.log(`   Projects:    GET/POST /api/projects`);
console.log(`   Graph:       GET /api/projects/:id/graph`);
console.log(`   Delete:      DELETE /api/projects/:id`);
