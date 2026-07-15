const routes = {
  research: {
    metrics: [
      ["pattern", "fusion"],
      ["orchestrator", "ollama:glm-5.2"],
      ["quorum", "2 / 3"],
      ["latency", "según proveedor"],
      ["estimated cost", "$0.1579"],
    ],
    candidates: [
      ["ollama:glm-5.2", "orchestrator"],
      ["google:gemini-3.5-flash", "selected"],
      ["openai:gpt-5.5", "selected"],
      ["anthropic:claude-sonnet-4-6", "selected"],
      ["anthropic:claude-opus-4-8", "selected"],
      ["ollama:deepseek-v4-pro", "available"],
      ["ollama:qwen3.5:122b", "available"],
      ["ollama:gpt-oss:120b", "available"],
    ],
    nodes: [
      ["request", 7, 50, "request", "Task", "Compare architectures", "research"],
      ["router", 23, 50, "router", "Orchestrator", "ollama:glm-5.2", "validated plan"],
      ["gemini", 45, 18, "success", "Panel / success", "gemini-3.5-flash", "independent analysis"],
      ["gpt-panel", 45, 50, "failure", "Panel / failed", "openai:gpt-5.5", "empty × 2"],
      ["claude", 45, 82, "success", "Panel / success", "claude-sonnet-4-6", "independent analysis"],
      ["judge", 68, 50, "review", "Judge", "claude-opus-4-8", "consensus + gaps"],
      ["writer", 90, 50, "output", "Final writer", "openai:gpt-5.5", "user answer"],
    ],
    edges: [
      ["request", "router"], ["router", "gemini"], ["router", "gpt-panel"], ["router", "claude"],
      ["gemini", "judge"], ["gpt-panel", "judge", "failed"], ["claude", "judge"], ["judge", "writer"],
    ],
  },
  structured: {
    metrics: [
      ["pattern", "cascade"],
      ["orchestrator", "openai:gpt-5.4-mini"],
      ["schema", "exact match"],
      ["latency", "según proveedor"],
      ["estimated cost", "$0.0198"],
    ],
    candidates: [
      ["openai:gpt-5.4-mini", "orchestrator"],
      ["anthropic:claude-sonnet-4-6", "selected"],
      ["google:gemini-3.1-flash-lite", "selected"],
      ["anthropic:claude-opus-4-8", "fallback"],
      ["ollama:qwen3.5:122b", "available"],
      ["ollama:deepseek-v4-pro", "available"],
      ["ollama:gpt-oss:120b", "available"],
      ["openai:gpt-5.4-nano", "available"],
    ],
    nodes: [
      ["request", 8, 50, "request", "Request", "Claim extraction", "JSON schema"],
      ["router", 27, 50, "router", "Orchestrator", "openai:gpt-5.4-mini", "cascade"],
      ["primary", 49, 50, "success", "Primary", "claude-sonnet-4-6", "typed JSON"],
      ["validator", 69, 28, "review", "Validator", "google:gemini-3.1-flash-lite", "schema + sufficiency"],
      ["escalation", 69, 73, "skipped", "Escalation / skipped", "claude-opus-4-8", "not called"],
      ["output", 90, 50, "output", "Result", "CLM-2048", "validated"],
    ],
    edges: [
      ["request", "router"], ["router", "primary"], ["primary", "validator"],
      ["validator", "output"], ["validator", "escalation", "skipped"], ["escalation", "output", "skipped"],
    ],
  },
  agentic: {
    metrics: [
      ["pattern", "critique_repair"],
      ["orchestrator", "google:gemini-3.5-flash"],
      ["model calls", "4"],
      ["latency", "según proveedor"],
      ["estimated cost", "$0.1072"],
    ],
    candidates: [
      ["google:gemini-3.5-flash", "orchestrator"],
      ["anthropic:claude-opus-4-8", "selected"],
      ["openai:gpt-5.5", "selected"],
      ["ollama:deepseek-v4-pro", "selected"],
      ["ollama:kimi-k2.7-code", "fallback"],
      ["ollama:qwen3-coder-next", "available"],
      ["ollama:minimax-m3", "available"],
      ["anthropic:claude-sonnet-4-6", "available"],
    ],
    nodes: [
      ["request", 7, 50, "request", "Request", "Payment retry review", "high risk"],
      ["router", 23, 50, "router", "Orchestrator", "google:gemini-3.5-flash", "critique_repair"],
      ["draft", 45, 50, "success", "Generator", "claude-opus-4-8", "merge decision"],
      ["critic", 67, 25, "failure", "Independent critic", "openai:gpt-5.5", "challenge draft"],
      ["repair", 67, 75, "review", "Repair", "ollama:deepseek-v4-pro", "apply critique"],
      ["output", 90, 50, "output", "Final", "direct answer", "internal notes removed"],
    ],
    edges: [
      ["request", "router"], ["router", "draft"], ["draft", "critic"],
      ["draft", "repair"], ["critic", "repair"], ["repair", "output"],
    ],
  },
  tools: {
    metrics: [
      ["pattern", "critique_repair"],
      ["tool rounds", "2 max"],
      ["tool calls", "1"],
      ["latency", "según proveedor"],
      ["estimated cost", "$0.0503"],
    ],
    candidates: [
      ["anthropic:claude-sonnet-4-6", "orchestrator"],
      ["ollama:kimi-k2.7-code", "selected"],
      ["ollama:qwen3-coder-next", "selected"],
      ["anthropic:claude-opus-4-8", "fallback"],
      ["openai:gpt-5.4-nano", "available"],
      ["google:gemini-3.1-flash-lite", "available"],
      ["ollama:deepseek-v4-pro", "available"],
    ],
    nodes: [
      ["request", 7, 50, "request", "Request", "Billing case reply", "SUP-LIVE-TOOL-1"],
      ["router", 22, 50, "router", "Orchestrator", "claude-sonnet-4-6", "tool-aware route"],
      ["planner", 40, 25, "success", "Tool planner", "ollama:kimi-k2.7-code", "approved call"],
      ["tool", 40, 75, "tool", "Local tool", "lookup_billing_case", "authoritative ledger"],
      ["critic", 64, 25, "review", "Tool critic", "claude-sonnet-4-6", "verify claims"],
      ["repair", 64, 75, "success", "Tool repair", "ollama:qwen3-coder-next", "remove unsupported"],
      ["output", 90, 50, "output", "Final", "Customer reply", "state + ETA"],
    ],
    edges: [
      ["request", "router"], ["router", "planner"], ["planner", "tool"], ["tool", "critic"],
      ["tool", "repair"], ["critic", "repair"], ["repair", "output"],
    ],
  },
  pdf: {
    metrics: [
      ["pattern", "single"],
      ["transport", "native PDF"],
      ["capability", "document input"],
      ["latency", "según proveedor"],
      ["estimated cost", "$0.0101"],
    ],
    candidates: [
      ["ollama:qwen3.5:122b", "orchestrator"],
      ["openai:gpt-5.4-mini", "selected"],
      ["openai:gpt-5.5", "fallback"],
      ["google:gemini-2.5-pro", "filtered"],
      ["anthropic:claude-sonnet-4-6", "filtered"],
      ["ollama:kimi-k2.6", "filtered"],
      ["ollama:deepseek-v4-pro", "filtered"],
    ],
    nodes: [
      ["request", 9, 50, "request", "File input", "audit.pdf", "application/pdf"],
      ["router", 31, 50, "router", "Orchestrator", "ollama:qwen3.5:122b", "capability filter"],
      ["model", 59, 50, "success", "Primary", "openai:gpt-5.4-mini", "native file"],
      ["output", 88, 50, "output", "Result", "zircon", "exact passphrase"],
    ],
    edges: [["request", "router"], ["router", "model"], ["model", "output"]],
  },
  coding: {
    metrics: [
      ["pattern", "panel_fusion"],
      ["orchestrator", "ollama:glm-5.2"],
      ["panel", "3 specialists"],
      ["capability", "tools + thinking"],
      ["transport", "multi-provider"],
    ],
    candidates: [
      ["ollama:glm-5.2", "orchestrator"],
      ["ollama:qwen3-coder-next", "selected"],
      ["ollama:deepseek-v4-pro", "selected"],
      ["ollama:kimi-k2.7-code", "selected"],
      ["ollama:minimax-m2.7", "selected"],
      ["anthropic:claude-sonnet-4-6", "selected"],
      ["openai:gpt-5.5", "fallback"],
      ["google:gemini-3.5-flash", "available"],
      ["ollama:gpt-oss:120b", "available"],
    ],
    nodes: [
      ["request", 7, 50, "request", "Task", "Refactor monorepo", "long-horizon code"],
      ["router", 23, 50, "router", "Orchestrator", "ollama:glm-5.2", "coding candidate set"],
      ["qwen", 45, 18, "success", "Code analyst", "ollama:qwen3-coder-next", "repo exploration"],
      ["deepseek", 45, 50, "success", "Reasoning peer", "ollama:deepseek-v4-pro", "migration risks"],
      ["kimi", 45, 82, "success", "Implementation peer", "ollama:kimi-k2.7-code", "patch strategy"],
      ["judge", 68, 50, "review", "Judge", "ollama:minimax-m2.7", "merge + constraints"],
      ["output", 90, 50, "output", "Final writer", "anthropic:claude-sonnet-4-6", "validated plan"],
    ],
    edges: [
      ["request", "router"], ["router", "qwen"], ["router", "deepseek"], ["router", "kimi"],
      ["qwen", "judge"], ["deepseek", "judge"], ["kimi", "judge"], ["judge", "output"],
    ],
  },
  vision: {
    metrics: [
      ["pattern", "multimodal_fusion"],
      ["orchestrator", "openai:gpt-5.4-mini"],
      ["input", "image"],
      ["capability", "vision + tools"],
      ["transport", "multi-provider"],
    ],
    candidates: [
      ["openai:gpt-5.4-mini", "orchestrator"],
      ["ollama:qwen3.5", "selected"],
      ["ollama:gemma4", "selected"],
      ["ollama:kimi-k2.6", "selected"],
      ["ollama:minimax-m3", "selected"],
      ["google:gemini-3.5-flash", "selected"],
      ["anthropic:claude-sonnet-4-6", "fallback"],
      ["ollama:mistral-large-3:675b", "available"],
    ],
    nodes: [
      ["request", 7, 50, "request", "Image input", "interface.png", "visual QA"],
      ["router", 23, 50, "router", "Orchestrator", "openai:gpt-5.4-mini", "vision allowlist"],
      ["qwen", 45, 18, "success", "Vision peer", "ollama:qwen3.5", "layout analysis"],
      ["gemma", 45, 50, "success", "Vision peer", "ollama:gemma4", "content + audio ready"],
      ["kimi", 45, 82, "success", "Agentic peer", "ollama:kimi-k2.6", "interaction risks"],
      ["judge", 68, 50, "review", "Fusion", "ollama:minimax-m3", "cross-modal synthesis"],
      ["output", 90, 50, "output", "Final writer", "google:gemini-3.5-flash", "prioritized findings"],
    ],
    edges: [
      ["request", "router"], ["router", "qwen"], ["router", "gemma"], ["router", "kimi"],
      ["qwen", "judge"], ["gemma", "judge"], ["kimi", "judge"], ["judge", "output"],
    ],
  },
  openSet: {
    metrics: [
      ["pattern", "critique_repair"],
      ["orchestrator", "ollama:gpt-oss:120b"],
      ["roles", "planner + critic"],
      ["capability", "agentic reasoning"],
      ["transport", "Ollama Cloud"],
    ],
    candidates: [
      ["ollama:gpt-oss:120b", "orchestrator"],
      ["ollama:nemotron-3-super", "selected"],
      ["ollama:mistral-large-3", "selected"],
      ["ollama:minimax-m2.7", "selected"],
      ["ollama:qwen3-coder:480b", "selected"],
      ["ollama:deepseek-v4-pro", "fallback"],
      ["ollama:glm-5.2", "available"],
      ["ollama:granite4.1", "available"],
    ],
    nodes: [
      ["request", 7, 50, "request", "Task", "Migration runbook", "multi-agent ops"],
      ["router", 23, 50, "router", "Orchestrator", "ollama:gpt-oss:120b", "open model route"],
      ["planner", 45, 50, "success", "Planner", "ollama:nemotron-3-super", "dependency graph"],
      ["critic", 67, 25, "review", "Independent critic", "ollama:mistral-large-3", "failure modes"],
      ["repair", 67, 75, "success", "Repair", "ollama:minimax-m2.7", "rollback gates"],
      ["output", 90, 50, "output", "Final writer", "ollama:qwen3-coder:480b", "ordered controls"],
    ],
    edges: [
      ["request", "router"], ["router", "planner"], ["planner", "critic"],
      ["planner", "repair"], ["critic", "repair"], ["repair", "output"],
    ],
  },
};

const codeSamples = {
  sdk: {
    filename: "app.py",
    code: `from crupier import Crupier

crupier = Crupier.from_project(".")

result = crupier.deal(
    task="Compare two production architectures.",
    input={"availability_target": "99.9%"},
    mode="research",
    constraints={
        "max_cost_usd": 0.50,
        "max_calls": 12,
        "min_panel_size": 3,
    },
    trace="debug",
    dry_run=False,
)

print(result.route.strategy)
print(result.route.model_summary)
print(result.output_text)`,
  },
  compat: {
    filename: "existing_client.py",
    code: `from crupier.compat.openai import OpenAI

client = OpenAI(project=".")

response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "user", "content": "Summarize this"}
    ],
)

print(response.choices[0].message.content)`,
  },
  server: {
    filename: "terminal.sh",
    code: `crupier serve --port 8787 --no-dry-run

export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"

curl "$OPENAI_BASE_URL/responses" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gpt-5.4-mini",
    "input": "Return one deployment risk"
  }'`,
  },
  cli: {
    filename: "terminal.sh",
    code: `crupier init
crupier models discover --provider openai
crupier capabilities readiness --strict

crupier route \\
  "Compare two implementation plans" \\
  --mode research

crupier deal \\
  "Write a concise support reply" \\
  --mode fast \\
  --no-dry-run`,
  },
};

const routeNodes = document.querySelector("#route-nodes");
const routeMetrics = document.querySelector("#route-metrics");
const modelPool = document.querySelector("#model-pool");
const graphStage = document.querySelector("#graph-stage");
const routeLab = document.querySelector(".route-lab");
const canvas = document.querySelector("#route-canvas");
const context = canvas.getContext("2d");
const routeOrder = Object.keys(routes);
const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
const ROUTE_CYCLE_MS = 5200;
let activeRoute = "research";
let toastTimer;
let routeCycleTimer;
let routeCyclePaused = false;

function renderRoute(routeName) {
  activeRoute = routeName;
  const route = routes[routeName];
  routeNodes.replaceChildren();
  routeMetrics.replaceChildren();
  modelPool.replaceChildren();

  route.candidates.forEach(([model, state]) => {
    const item = document.createElement("li");
    const modelElement = document.createElement("strong");
    const stateElement = document.createElement("small");
    item.className = `is-${state}`;
    modelElement.textContent = model;
    stateElement.textContent = state;
    item.append(modelElement, stateElement);
    modelPool.append(item);
  });

  route.nodes.forEach(([id, x, y, state, role, model, note], index) => {
    const node = document.createElement("li");
    node.className = `route-node is-${state}`;
    node.dataset.node = id;
    node.dataset.column = routeColumn(x);
    node.style.setProperty("--x", `${x}%`);
    node.style.setProperty("--y", `${y}%`);
    node.style.setProperty("--delay", `${index * 45}ms`);
    const roleElement = document.createElement("small");
    const modelElement = document.createElement("strong");
    const noteElement = document.createElement("span");
    roleElement.textContent = role;
    modelElement.textContent = model;
    noteElement.textContent = note;
    node.append(roleElement, modelElement, noteElement);
    routeNodes.append(node);
  });

  const lastNode = routeNodes.lastElementChild;
  if (lastNode && !motionPreference.matches) {
    lastNode.addEventListener("animationend", () => requestAnimationFrame(drawEdges), { once: true });
  }

  route.metrics.forEach(([term, value], index) => {
    const item = document.createElement("div");
    item.style.setProperty("--delay", `${index * 35}ms`);
    const termElement = document.createElement("dt");
    const valueElement = document.createElement("dd");
    termElement.textContent = term;
    valueElement.textContent = value;
    item.append(termElement, valueElement);
    routeMetrics.append(item);
  });

  document.querySelectorAll("[data-route]").forEach((button) => {
    const isActive = button.dataset.route === routeName;
    button.setAttribute("aria-selected", String(isActive));
    button.tabIndex = isActive ? 0 : -1;
    if (isActive) {
      document.querySelector("#route-panel").setAttribute("aria-labelledby", button.id);
    }
  });

  requestAnimationFrame(drawEdges);
}

function routeColumn(x) {
  if (x <= 12) return "start";
  if (x < 35) return "router";
  if (x < 58) return "workers";
  if (x < 80) return "review";
  return "output";
}

function resizeRouteLab() {
  drawEdges();
}

function drawEdges() {
  const rect = graphStage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const canvasWidth = Math.round(rect.width * ratio);
  const canvasHeight = Math.round(rect.height * ratio);
  if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  context.save();
  context.beginPath();
  for (let x = 40; x < rect.width; x += 40) {
    context.moveTo(x, 0);
    context.lineTo(x, rect.height);
  }
  for (let y = 40; y < rect.height; y += 40) {
    context.moveTo(0, y);
    context.lineTo(rect.width, y);
  }
  context.lineWidth = 1;
  context.strokeStyle = "#cbc9bf";
  context.stroke();
  context.restore();

  const edges = routes[activeRoute].edges.flatMap(([fromId, toId, state], index) => {
    const from = routeNodes.querySelector(`[data-node="${fromId}"]`);
    const to = routeNodes.querySelector(`[data-node="${toId}"]`);
    if (!from || !to) return [];

    const fromRect = from.getBoundingClientRect();
    const toRect = to.getBoundingClientRect();
    const fromCenterX = fromRect.left + fromRect.width / 2;
    const fromCenterY = fromRect.top + fromRect.height / 2;
    const toCenterX = toRect.left + toRect.width / 2;
    const toCenterY = toRect.top + toRect.height / 2;
    return [{
      index,
      fromId,
      toId,
      state,
      fromRect,
      toRect,
      fromCenterX,
      fromCenterY,
      toCenterX,
      toCenterY,
      vertical: Math.abs(fromCenterX - toCenterX) < 16 && toRect.top >= fromRect.bottom - 4,
    }];
  });

  const horizontalEdges = edges.filter((edge) => !edge.vertical);
  const outgoing = groupEdges(horizontalEdges, "fromId", "toCenterY");
  const incoming = groupEdges(horizontalEdges, "toId", "fromCenterY");

  edges.forEach((edge) => {
    let startX;
    let startY;
    let endX;
    let endY;
    let control;

    if (edge.vertical) {
      startX = edge.fromCenterX - rect.left;
      startY = edge.fromRect.bottom - rect.top;
      endX = edge.toCenterX - rect.left;
      endY = edge.toRect.top - rect.top;
      control = Math.max(20, (endY - startY) * 0.5);
    } else {
      startX = edge.fromRect.right - rect.left;
      startY = edge.fromRect.top + edge.fromRect.height * edgePort(outgoing.get(edge.fromId), edge) - rect.top;
      endX = edge.toRect.left - rect.left;
      endY = edge.toRect.top + edge.toRect.height * edgePort(incoming.get(edge.toId), edge) - rect.top;
      control = Math.max(28, (endX - startX) * 0.46);
    }

    context.save();
    context.beginPath();
    context.moveTo(startX, startY);
    if (edge.vertical) {
      context.bezierCurveTo(startX, startY + control, endX, endY - control, endX, endY);
    } else {
      context.bezierCurveTo(startX + control, startY, endX - control, endY, endX, endY);
    }
    context.lineWidth = 1.5;
    context.strokeStyle = edge.state === "failed" ? "#d94f3d" : "#141414";
    if (edge.state === "skipped") context.setLineDash([7, 6]);
    context.stroke();

    context.beginPath();
    context.arc(endX, endY, 3, 0, Math.PI * 2);
    context.fillStyle = edge.state === "failed" ? "#d94f3d" : "#141414";
    context.fill();
    context.restore();
  });
}

function groupEdges(edges, groupKey, sortKey) {
  const groups = new Map();
  edges.forEach((edge) => {
    const key = edge[groupKey];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(edge);
  });
  groups.forEach((group) => group.sort((left, right) => left[sortKey] - right[sortKey] || left.index - right.index));
  return groups;
}

function edgePort(group, edge) {
  if (!group || group.length === 1) return 0.5;
  return (group.indexOf(edge) + 1) / (group.length + 1);
}

function renderCode(name) {
  const sample = codeSamples[name];
  document.querySelector("#code-filename").textContent = sample.filename;
  document.querySelector("#code-sample").textContent = sample.code;
  document.querySelectorAll("[data-code]").forEach((button) => {
    const isActive = button.dataset.code === name;
    button.setAttribute("aria-selected", String(isActive));
    button.tabIndex = isActive ? 0 : -1;
    if (isActive) {
      document.querySelector("#code-panel").setAttribute("aria-labelledby", button.id);
    }
  });
}

function bindTabs(selector, dataKey, render) {
  const tabs = [...document.querySelectorAll(selector)];
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => render(tab.dataset[dataKey]));
    tab.addEventListener("keydown", (event) => {
      const current = tabs.indexOf(tab);
      let next = current;
      if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === current) return;
      event.preventDefault();
      tabs[next].focus();
      tabs[next].click();
    });
  });
}

function stopRouteCycle() {
  window.clearTimeout(routeCycleTimer);
  routeCycleTimer = undefined;
  routeLab.classList.remove("is-cycling");
}

function scheduleRouteCycle() {
  stopRouteCycle();
  if (routeCyclePaused || document.hidden || motionPreference.matches) return;
  void routeLab.offsetWidth;
  routeLab.classList.add("is-cycling");
  routeCycleTimer = window.setTimeout(() => {
    const current = routeOrder.indexOf(activeRoute);
    renderRoute(routeOrder[(current + 1) % routeOrder.length]);
    scheduleRouteCycle();
  }, ROUTE_CYCLE_MS);
}

function pauseRouteCycle() {
  routeCyclePaused = true;
  stopRouteCycle();
}

function resumeRouteCycle() {
  routeCyclePaused = routeLab.matches(":hover") || routeLab.contains(document.activeElement);
  if (!routeCyclePaused) scheduleRouteCycle();
}

function copyWithSelection(text) {
  const input = document.createElement("textarea");
  input.value = text;
  input.setAttribute("readonly", "");
  input.style.position = "fixed";
  input.style.opacity = "0";
  document.body.append(input);
  input.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } finally {
    input.remove();
  }
  return copied;
}

async function copyText(text) {
  showToast("Copiando");
  const selectionCopied = copyWithSelection(text);
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
    await Promise.race([
      navigator.clipboard.writeText(text),
      new Promise((_, reject) => window.setTimeout(() => reject(new Error("Clipboard timeout")), 1000)),
    ]);
    showToast("Copiado");
  } catch {
    showToast(selectionCopied ? "Copiado" : "No se pudo copiar");
  }
}

function showToast(message) {
  const toast = document.querySelector("#copy-toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 1400);
}

bindTabs("[data-route]", "route", (routeName) => {
  renderRoute(routeName);
  scheduleRouteCycle();
});
bindTabs("[data-code]", "code", renderCode);

routeLab.addEventListener("pointerenter", pauseRouteCycle);
routeLab.addEventListener("pointerleave", resumeRouteCycle);
routeLab.addEventListener("focusin", pauseRouteCycle);
routeLab.addEventListener("focusout", () => window.setTimeout(resumeRouteCycle, 0));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopRouteCycle();
  else resumeRouteCycle();
});
motionPreference.addEventListener("change", () => {
  if (motionPreference.matches) stopRouteCycle();
  else resumeRouteCycle();
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", () => copyText(button.dataset.copy));
});

document.querySelector("#copy-code").addEventListener("click", () => {
  copyText(document.querySelector("#code-sample").textContent);
});

window.addEventListener("resize", resizeRouteLab);

renderRoute(activeRoute);
renderCode("sdk");
scheduleRouteCycle();
