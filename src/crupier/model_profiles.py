"""Decision metadata for model routing.

The selector should not rely on model-name vibes alone. This module enriches
capability cards with a compact, evidence-labeled decision profile that can be
used by both deterministic scoring and the opt-in LLM orchestrator.
"""

from __future__ import annotations

import re
from typing import Any

from .models import CapabilityCard, RequestEnvelope

SOURCE_SNAPSHOT_DATE = "2026-06-20"

SOURCE_URLS = {
    "openai_models": "https://developers.openai.com/api/docs/models",
    "openai_deprecations": "https://developers.openai.com/api/docs/deprecations",
    "openai_gpt_5_5": "https://developers.openai.com/api/docs/models/gpt-5.5",
    "anthropic_models": "https://platform.claude.com/docs/en/about-claude/models/overview",
    "anthropic_opus_4_8": "https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8",
    "google_models": "https://ai.google.dev/gemini-api/docs/models",
    "google_gemini_3_5_flash": "https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash",
    "google_gemini_3_1_flash_lite": "https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite",
    "google_imagen": "https://ai.google.dev/gemini-api/docs/models/imagen",
    "google_veo_2": "https://ai.google.dev/gemini-api/docs/models/veo-2.0-generate-001",
    "ollama_cloud": "https://docs.ollama.com/cloud",
    "arena_text": "https://arena.ai/leaderboard/text",
    "artificial_analysis": "https://artificialanalysis.ai/",
    "swe_bench": "https://www.swebench.com/",
    "livebench": "https://livebench.ai/",
}

TASK_KEYWORDS = {
    "agentic": [
        "agent",
        "agentic",
        "agente",
        "agentes",
        "autonomous",
        "autonomo",
        "autonomo",
        "workflow",
        "multi-step",
        "multistep",
        "long-horizon",
        "sistema",
        "automatizar",
    ],
    "coding": [
        "code",
        "coding",
        "coder",
        "repo",
        "github",
        "bug",
        "debug",
        "refactor",
        "test",
        "patch",
        "codigo",
        "código",
        "repositorio",
        "depurar",
        "arreglar",
        "prueba",
    ],
    "console": [
        "terminal",
        "shell",
        "console",
        "cli",
        "command",
        "comando",
        "consola",
        "bash",
        "zsh",
    ],
    "critique": [
        "review",
        "critique",
        "audit",
        "compare",
        "risk",
        "revisar",
        "criticar",
        "auditar",
        "comparar",
        "riesgo",
    ],
    "embeddings": [
        "embedding",
        "embeddings",
        "vector",
        "vectors",
        "semantic",
        "rag",
        "retrieval",
        "busqueda semantica",
        "búsqueda semántica",
        "vectores",
    ],
    "long_context": [
        "long context",
        "long-context",
        "large context",
        "codebase",
        "repo completo",
        "documentos largos",
        "muchos archivos",
        "large file",
    ],
    "math": [
        "math",
        "mathematics",
        "calculus",
        "proof",
        "geometry",
        "algebra",
        "matematicas",
        "matemáticas",
        "ecuacion",
        "ecuación",
        "demuestra",
    ],
    "multimodal": [
        "image",
        "images",
        "video",
        "audio",
        "vision",
        "multimodal",
        "imagen",
        "imagenes",
        "imágenes",
        "foto",
        "video",
        "audio",
    ],
    "pdf": [
        "pdf",
        "document",
        "documento",
        "contract",
        "contrato",
        "invoice",
        "factura",
        "receipt",
        "recibo",
        "ocr",
        "scan",
        "scanned",
        "escaneado",
    ],
    "privacy": [
        "private",
        "privacy",
        "local",
        "pii",
        "secret",
        "confidential",
        "privado",
        "privacidad",
        "secreto",
        "confidencial",
        "datos sensibles",
    ],
    "reasoning": [
        "reason",
        "reasoning",
        "logic",
        "plan",
        "strategy",
        "razonar",
        "razonamiento",
        "logica",
        "lógica",
        "estrategia",
        "planificar",
    ],
    "research": [
        "research",
        "sources",
        "cite",
        "citation",
        "investigar",
        "fuentes",
        "citas",
        "citar",
        "evidence",
        "evidencia",
    ],
    "structured_output": [
        "json",
        "schema",
        "structured",
        "extract",
        "parse",
        "csv",
        "fields",
        "estructura",
        "extrae",
        "extraer",
        "campos",
        "tabla",
    ],
    "low_cost": ["cheap", "cost", "budget", "barato", "coste", "costo", "presupuesto"],
    "low_latency": ["fast", "quick", "latency", "realtime", "rapido", "rápido", "latencia", "tiempo real"],
}

HUMAN_SKILL_LABELS = {
    "agentic": "agentic workflows",
    "coding": "coding and debugging",
    "console": "console and tool-heavy work",
    "critique": "critique and review",
    "document_extraction": "document extraction",
    "embeddings": "embeddings and retrieval",
    "long_context": "long-context work",
    "math": "math and formal reasoning",
    "multimodal": "multimodal understanding",
    "orchestration": "model routing and orchestration",
    "pdf": "PDF and document understanding",
    "privacy": "private or local-first routing",
    "reasoning": "complex reasoning",
    "research": "research and synthesis",
    "structured_output": "structured output",
    "tool_use": "tool use",
    "vision": "image understanding",
    "low_cost": "low-cost workloads",
    "low_latency": "low-latency workloads",
}

OFFICIAL_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "openai:gpt-5.5": {
        "context_window": 1_050_000,
        "max_output_tokens": 128_000,
        "modalities_input": ["text", "image"],
        "modalities_output": ["text"],
        "pricing": {
            "input_per_million_usd": 5.0,
            "output_per_million_usd": 30.0,
            "cached_input_per_million_usd": 0.5,
            "confidence": "official",
            "source": SOURCE_URLS["openai_gpt_5_5"],
        },
        "latency_tier": "fast",
        "cost_tier": "high",
        "quality_tier": "frontier",
        "strengths": ["reasoning", "coding", "agentic", "quality", "tool_use", "structured_output", "vision"],
        "skill_scores": {
            "reasoning": 9.2,
            "coding": 9.2,
            "agentic": 8.8,
            "tool_use": 8.8,
            "structured_output": 8.6,
            "long_context": 9.0,
            "vision": 7.8,
        },
        "natural_summary": "Flagship OpenAI model for complex reasoning, coding, professional work, and tool-heavy agentic tasks.",
        "sources": ["openai_models", "openai_gpt_5_5"],
    },
    "openai:gpt-5.4": {
        "context_window": 1_050_000,
        "max_output_tokens": 128_000,
        "modalities_input": ["text", "image"],
        "pricing": {
            "input_per_million_usd": 2.5,
            "output_per_million_usd": 15.0,
            "confidence": "official",
            "source": SOURCE_URLS["openai_models"],
        },
        "latency_tier": "fast",
        "cost_tier": "medium",
        "quality_tier": "frontier",
        "strengths": ["reasoning", "coding", "agentic", "tool_use", "structured_output"],
        "skill_scores": {"reasoning": 8.8, "coding": 8.8, "agentic": 8.5, "tool_use": 8.6},
        "natural_summary": "Balanced OpenAI frontier model for coding and professional work at lower cost than GPT-5.5.",
        "sources": ["openai_models"],
    },
    "openai:gpt-5.4-mini": {
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "modalities_input": ["text", "image"],
        "pricing": {
            "input_per_million_usd": 0.75,
            "output_per_million_usd": 4.5,
            "confidence": "official",
            "source": SOURCE_URLS["openai_models"],
        },
        "latency_tier": "fast",
        "cost_tier": "low",
        "quality_tier": "strong",
        "strengths": ["low_latency", "low_cost", "coding", "agentic", "tool_use", "structured_output"],
        "skill_scores": {
            "low_latency": 9.0,
            "low_cost": 8.6,
            "coding": 8.1,
            "agentic": 8.0,
            "tool_use": 8.4,
            "structured_output": 8.4,
            "orchestration": 8.0,
        },
        "natural_summary": "Strong mini OpenAI model for low-cost coding, computer-use style subagents, and high-volume routing.",
        "sources": ["openai_models"],
    },
    "openai:gpt-5.4-nano": {
        "latency_tier": "fast",
        "cost_tier": "low",
        "quality_tier": "strong",
        "strengths": ["low_latency", "low_cost", "structured_output"],
        "skill_scores": {"low_latency": 9.2, "low_cost": 9.3, "structured_output": 7.8, "orchestration": 7.8},
        "natural_summary": "Cheapest GPT-5.4-class option for simple high-volume tasks and lightweight classification.",
        "sources": ["openai_models"],
    },
    "anthropic:claude-opus-4-8": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "modalities_input": ["text", "image", "pdf", "file"],
        "pricing": {
            "input_per_million_usd": 5.0,
            "output_per_million_usd": 25.0,
            "confidence": "official",
            "source": SOURCE_URLS["anthropic_models"],
        },
        "cost_tier": "high",
        "latency_tier": "medium",
        "quality_tier": "frontier",
        "strengths": ["reasoning", "coding", "agentic", "critique", "quality", "tool_use", "structured_output"],
        "skill_scores": {
            "reasoning": 9.5,
            "coding": 9.5,
            "agentic": 9.5,
            "critique": 9.2,
            "tool_use": 8.8,
            "structured_output": 8.6,
            "pdf": 8.6,
            "long_context": 9.2,
            "orchestration": 9.0,
        },
        "natural_summary": "Anthropic Opus-tier model for complex reasoning, long-horizon agentic coding, and high-autonomy work.",
        "sources": ["anthropic_models", "anthropic_opus_4_8"],
    },
    "anthropic:claude-sonnet-4-6": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "modalities_input": ["text", "image", "pdf", "file"],
        "pricing": {
            "input_per_million_usd": 3.0,
            "output_per_million_usd": 15.0,
            "confidence": "official",
            "source": SOURCE_URLS["anthropic_models"],
        },
        "cost_tier": "medium",
        "latency_tier": "fast",
        "quality_tier": "frontier",
        "strengths": ["reasoning", "coding", "agentic", "critique", "quality", "tool_use", "structured_output"],
        "skill_scores": {
            "reasoning": 8.9,
            "coding": 9.0,
            "agentic": 9.0,
            "critique": 8.7,
            "tool_use": 8.8,
            "structured_output": 8.6,
            "pdf": 8.4,
            "long_context": 9.0,
            "orchestration": 8.6,
        },
        "natural_summary": "Frontier Claude model with strong speed, thinking, tool use, and long-context agentic reliability.",
        "sources": ["anthropic_models"],
    },
    "google:gemini-3.5-flash": {
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
        "modalities_input": ["text", "image", "video", "audio", "pdf"],
        "modalities_output": ["text"],
        "cost_tier": "low",
        "latency_tier": "fast",
        "quality_tier": "frontier",
        "strengths": ["low_latency", "low_cost", "agentic", "coding", "multimodal", "tool_use", "structured_output"],
        "skill_scores": {
            "agentic": 8.8,
            "coding": 8.8,
            "low_latency": 8.8,
            "low_cost": 8.0,
            "multimodal": 9.3,
            "pdf": 9.0,
            "structured_output": 8.8,
            "tool_use": 8.6,
            "long_context": 9.0,
            "orchestration": 8.5,
        },
        "natural_summary": "Fast frontier Gemini model for multimodal, long-context, coding, and agentic loops at strong value.",
        "sources": ["google_gemini_3_5_flash"],
    },
    "google:gemini-3.1-flash-lite": {
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
        "modalities_input": ["text", "image", "video", "audio", "pdf"],
        "modalities_output": ["text"],
        "cost_tier": "low",
        "latency_tier": "fast",
        "quality_tier": "strong",
        "strengths": ["low_latency", "low_cost", "multimodal", "structured_output", "agentic"],
        "skill_scores": {
            "low_latency": 9.4,
            "low_cost": 9.3,
            "multimodal": 8.5,
            "pdf": 8.7,
            "structured_output": 8.2,
            "document_extraction": 8.8,
            "orchestration": 9.0,
        },
        "natural_summary": "Low-latency, low-cost multimodal Gemini model for classification, routing, extraction, and PDF triage.",
        "sources": ["google_gemini_3_1_flash_lite"],
    },
    "google:gemini-3.1-pro-preview": {
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
        "modalities_input": ["text", "image", "video", "audio", "pdf"],
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["reasoning", "coding", "agentic", "tool_use", "multimodal", "structured_output"],
        "skill_scores": {"reasoning": 8.9, "coding": 8.8, "agentic": 8.8, "tool_use": 8.8, "multimodal": 8.7},
        "natural_summary": "Gemini Pro preview model for precise tool usage, software engineering behavior, and agentic workflows.",
        "sources": ["google_gemini_3_5_flash", "google_gemini_3_1_flash_lite"],
    },
}

PRODUCTION_RECOMMENDED_MODELS = {
    "openai:gpt-5.5",
    "openai:gpt-5.4",
    "openai:gpt-5.4-mini",
    "openai:gpt-5.4-nano",
    "anthropic:claude-opus-4-8",
    "anthropic:claude-sonnet-4-6",
    "google:gemini-3.5-flash",
    "google:gemini-3.1-flash-lite",
    "google:gemini-2.5-pro",
    "google:gemini-2.5-flash",
    "google:gemini-2.5-flash-lite",
    "ollama:glm-5.2",
    "ollama:gpt-oss:120b",
}

OPT_IN_RECOMMENDED_MODELS = {
    "google:gemini-3.1-pro-preview",
    "google:deep-research-max-preview-04-2026",
    "google:deep-research-preview-04-2026",
    "google:antigravity-preview-05-2026",
    "ollama:gemini-3-flash-preview",
}

EXPENSIVE_OPT_IN_MODELS = {
    "openai:o3",
    "openai:o3-pro",
    "openai:o4-mini",
}

CURATED_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "google:gemini-2.5-pro": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["reasoning", "coding", "agentic", "multimodal", "tool_use", "structured_output"],
        "skill_scores": {
            "reasoning": 8.6,
            "coding": 8.4,
            "agentic": 8.4,
            "multimodal": 8.6,
            "structured_output": 8.4,
            "tool_use": 8.2,
            "long_context": 8.6,
        },
        "natural_summary": "Gemini Pro model for harder reasoning, coding, long-context analysis, and multimodal work when Flash quality is not enough.",
        "sources": ["google_models"],
    },
    "google:gemini-2.5-flash": {
        "quality_tier": "strong",
        "cost_tier": "low",
        "latency_tier": "fast",
        "strengths": ["low_latency", "low_cost", "multimodal", "tool_use", "structured_output"],
        "skill_scores": {
            "low_latency": 8.8,
            "low_cost": 8.4,
            "multimodal": 8.4,
            "structured_output": 8.2,
            "tool_use": 8.0,
            "pdf": 8.2,
        },
        "natural_summary": "Fast, lower-cost Gemini model for multimodal classification, extraction, summarization, and everyday agent turns.",
        "sources": ["google_models"],
    },
    "google:gemini-2.5-flash-lite": {
        "quality_tier": "strong",
        "cost_tier": "low",
        "latency_tier": "fast",
        "strengths": ["low_latency", "low_cost", "multimodal", "structured_output"],
        "skill_scores": {
            "low_latency": 9.0,
            "low_cost": 9.0,
            "multimodal": 8.0,
            "structured_output": 8.0,
            "document_extraction": 8.0,
        },
        "natural_summary": "Lowest-cost Gemini Flash-family option for high-volume routing, classification, extraction, and lightweight multimodal tasks.",
        "sources": ["google_models"],
    },
    "ollama:glm-5.2": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["reasoning", "agentic", "coding", "orchestration", "privacy"],
        "skill_scores": {"reasoning": 8.6, "agentic": 8.6, "coding": 8.2, "orchestration": 8.6, "privacy": 8.0},
        "natural_summary": "Ollama Cloud GLM model for orchestration, reasoning, coding plans, and long-horizon agent decisions when the configured account has access.",
        "sources": ["ollama_cloud"],
    },
    "ollama:gpt-oss:120b": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["reasoning", "coding", "quality", "privacy"],
        "skill_scores": {"reasoning": 8.4, "coding": 8.2, "privacy": 8.0},
        "natural_summary": "Large Ollama Cloud text model for general reasoning and coding-style analysis when an account-local Cloud model is preferred.",
        "sources": ["ollama_cloud"],
    },
    "ollama:deepseek-v4-pro": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["reasoning", "coding", "agentic"],
    },
    "ollama:kimi-k2.7-code": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["coding", "agentic", "tool_use"],
    },
    "ollama:minimax-m3": {
        "quality_tier": "strong",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["agentic", "coding", "multimodal"],
    },
    "ollama:mistral-large-3:675b": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["reasoning", "quality"],
    },
    "ollama:qwen3-coder-next": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["coding", "agentic"],
    },
    "ollama:qwen3-coder:480b": {
        "quality_tier": "frontier",
        "cost_tier": "medium",
        "latency_tier": "medium",
        "strengths": ["coding", "agentic"],
    },
}

SPECIALIZED_MODEL_USES: dict[str, dict[str, Any]] = {
    "openai:text-embedding-3-small": {
        "model_kind": "embedding",
        "status_reason": "Current OpenAI embedding model; route only retrieval/vector workloads here.",
        "preferred_when": ["embeddings", "rag"],
    },
    "openai:text-embedding-3-large": {
        "model_kind": "embedding",
        "status_reason": "Higher-quality OpenAI embedding model; route only retrieval/vector workloads here.",
        "preferred_when": ["embeddings", "rag"],
    },
    "openai:gpt-image-2": {
        "model_kind": "image",
        "modalities_output": ["image"],
        "status_reason": "Current OpenAI image generation model; not a default text-routing candidate.",
        "preferred_when": ["image_generation"],
    },
    "openai:gpt-audio-1.5": {
        "model_kind": "audio",
        "modalities_input": ["text", "audio"],
        "modalities_output": ["text", "audio"],
        "status_reason": "Current OpenAI audio model; use only for audio routes.",
        "preferred_when": ["audio"],
    },
    "openai:gpt-audio-mini": {
        "model_kind": "audio",
        "modalities_input": ["text", "audio"],
        "modalities_output": ["text", "audio"],
        "status_reason": "Lower-cost OpenAI audio model; use only for audio routes.",
        "preferred_when": ["audio", "low_cost"],
    },
    "openai:gpt-realtime-2": {
        "model_kind": "realtime",
        "modalities_input": ["text", "audio"],
        "modalities_output": ["text", "audio"],
        "status_reason": "OpenAI realtime model; use only for realtime voice/interactive routes.",
        "preferred_when": ["realtime", "audio", "low_latency"],
    },
    "openai:gpt-realtime-mini": {
        "model_kind": "realtime",
        "modalities_input": ["text", "audio"],
        "modalities_output": ["text", "audio"],
        "status_reason": "Lower-cost OpenAI realtime model; use only for realtime voice/interactive routes.",
        "preferred_when": ["realtime", "audio", "low_latency", "low_cost"],
    },
    "openai:whisper-1": {
        "model_kind": "transcription",
        "modalities_input": ["audio"],
        "modalities_output": ["text"],
        "status_reason": "Speech-to-text model; not a general chat route.",
        "preferred_when": ["transcription", "audio"],
    },
    "google:gemini-embedding-001": {
        "model_kind": "embedding",
        "status_reason": "Gemini embedding model; route only retrieval/vector workloads here.",
        "preferred_when": ["embeddings", "rag"],
    },
    "google:gemini-embedding-2": {
        "model_kind": "embedding",
        "status_reason": "Current Gemini multimodal embedding model; route only retrieval/vector workloads here.",
        "preferred_when": ["embeddings", "rag", "multimodal"],
    },
    "google:gemini-3.1-flash-image": {
        "model_kind": "image",
        "modalities_input": ["text", "image"],
        "modalities_output": ["text", "image"],
        "status_reason": "Current Gemini image generation/editing model; not a default text-routing candidate.",
        "preferred_when": ["image_generation", "vision"],
    },
    "google:gemini-3-pro-image": {
        "model_kind": "image",
        "modalities_input": ["text", "image"],
        "modalities_output": ["text", "image"],
        "status_reason": "Gemini Pro image model; not a default text-routing candidate.",
        "preferred_when": ["image_generation", "vision"],
    },
}

DEPRECATED_MODEL_NOTICES: dict[str, dict[str, Any]] = {
    "openai:gpt-5.2-chat-latest": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-08-10",
        "replacement": "openai:gpt-5.5",
        "source": "openai_deprecations",
    },
    "openai:gpt-5.3-chat-latest": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-08-10",
        "replacement": "openai:gpt-5.5",
        "source": "openai_deprecations",
    },
    "openai:chatgpt-image-latest": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-12-01",
        "replacement": "openai:gpt-image-2",
        "source": "openai_deprecations",
    },
    "openai:gpt-image-1-mini": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-12-01",
        "replacement": "openai:gpt-image-2",
        "source": "openai_deprecations",
    },
    "openai:gpt-image-1.5": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-12-01",
        "replacement": "openai:gpt-image-2",
        "source": "openai_deprecations",
    },
    "openai:sora-2": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-09-24",
        "replacement": None,
        "source": "openai_deprecations",
    },
    "openai:sora-2-pro": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-09-24",
        "replacement": None,
        "source": "openai_deprecations",
    },
    "google:gemini-2.0-flash": {
        "lifecycle": "shutdown",
        "replacement": "google:gemini-3.5-flash",
        "source": "google_models",
    },
    "google:gemini-2.0-flash-001": {
        "lifecycle": "shutdown",
        "replacement": "google:gemini-3.5-flash",
        "source": "google_models",
    },
    "google:gemini-2.0-flash-lite": {
        "lifecycle": "shutdown",
        "replacement": "google:gemini-3.1-flash-lite",
        "source": "google_models",
    },
    "google:gemini-2.0-flash-lite-001": {
        "lifecycle": "shutdown",
        "replacement": "google:gemini-3.1-flash-lite",
        "source": "google_models",
    },
    "google:gemini-3-pro-preview": {
        "lifecycle": "shutdown",
        "shutdown_date": "2026-03-09",
        "replacement": "google:gemini-3.1-pro-preview",
        "source": "google_models",
    },
    "google:gemini-3.1-flash-lite-preview": {
        "lifecycle": "shutdown",
        "replacement": "google:gemini-3.1-flash-lite",
        "source": "google_models",
    },
    "google:imagen-4.0-fast-generate-001": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-08-17",
        "replacement": "google:gemini-3.1-flash-image",
        "source": "google_imagen",
    },
    "google:imagen-4.0-generate-001": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-08-17",
        "replacement": "google:gemini-3.1-flash-image",
        "source": "google_imagen",
    },
    "google:imagen-4.0-ultra-generate-001": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-08-17",
        "replacement": "google:gemini-3.1-flash-image",
        "source": "google_imagen",
    },
    "google:veo-2.0-generate-001": {
        "lifecycle": "deprecated",
        "shutdown_date": "2026-06-30",
        "replacement": None,
        "source": "google_veo_2",
    },
}


def classify_task_signal_weights(request: RequestEnvelope) -> dict[str, float]:
    text_parts = [request.task]
    if isinstance(request.input, str):
        text_parts.append(request.input)
    for message in request.messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            text_parts.append(content)
    text = " ".join(text_parts).lower()

    signals: dict[str, float] = {}
    for signal, keywords in TASK_KEYWORDS.items():
        matches = _keyword_match_count(text, keywords)
        if matches:
            _boost_signal(signals, signal, min(1.0, 0.35 + (matches * 0.25)))

    mode = request.mode or ""
    if mode:
        _boost_signal(signals, mode, 0.7)
    if request.tools:
        _boost_signal(signals, "agentic", 1.0)
        _boost_signal(signals, "tool_use", 1.0)
    if request.response_schema is not None or request.constraints.get("response_schema"):
        _boost_signal(signals, "structured_output", 1.0)
    if request.files:
        _boost_signal(signals, "multimodal", 0.8)
    if request.file_plan is not None:
        for modality in request.file_plan.required_model_modalities:
            if modality == "image":
                _boost_signal(signals, "vision", 1.0)
                _boost_signal(signals, "multimodal", 1.0)
            elif modality in {"audio", "video"}:
                _boost_signal(signals, modality, 1.0)
                _boost_signal(signals, "multimodal", 1.0)
            elif modality == "file":
                _boost_signal(signals, "pdf", 0.8)
        for capability in request.file_plan.required_model_capabilities:
            if capability == "pdf_native_input":
                _boost_signal(signals, "pdf", 1.0)
                _boost_signal(signals, "document_extraction", 1.0)
            elif capability == "vision_input":
                _boost_signal(signals, "vision", 1.0)
                _boost_signal(signals, "multimodal", 1.0)
    if "pdf" in signals:
        _boost_signal(signals, "document_extraction", 0.9)
    if "console" in signals:
        _boost_signal(signals, "agentic", 0.75)
        _boost_signal(signals, "tool_use", 0.75)
    if "coding" in signals and mode == "agentic":
        _boost_signal(signals, "agentic", 0.85)
    return dict(sorted(signals.items()))


def classify_task_signals(request: RequestEnvelope) -> set[str]:
    return set(classify_task_signal_weights(request))


def _boost_signal(signals: dict[str, float], signal: str, value: float) -> None:
    signals[signal] = max(signals.get(signal, 0.0), min(1.0, max(0.0, float(value))))


def _keyword_match_count(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if _keyword_matches(text, keyword))


def _keyword_matches(text: str, keyword: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    if any(separator in keyword for separator in (" ", "-", "_")):
        return keyword in text
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def apply_decision_profile(card: CapabilityCard, *, provider_metadata: dict[str, Any] | None = None) -> CapabilityCard:
    provider_metadata = dict(provider_metadata or {})
    _apply_provider_metadata(card, provider_metadata)
    _apply_official_override(card)
    _apply_family_inference(card)
    _apply_lifecycle_profile(card)
    _apply_capability_status(card)
    card.skill_scores = _merged_skill_scores(card)
    card.natural_profile = _natural_profile(card)
    card.routing_hints = _routing_hints(card)
    card.evidence = _evidence(card, provider_metadata=provider_metadata)
    card.strengths = sorted(set(card.strengths))
    card.modalities_input = sorted(set(card.modalities_input), key=_modality_sort_key)
    card.modalities_output = sorted(set(card.modalities_output), key=_modality_sort_key)
    return card


def _apply_provider_metadata(card: CapabilityCard, metadata: dict[str, Any]) -> None:
    _apply_metadata_pricing(card, metadata)
    provider = card.model_ref.provider
    if provider == "anthropic":
        card.context_window = card.context_window or _int_or_none(metadata.get("max_input_tokens"))
        card.max_output_tokens = card.max_output_tokens or _int_or_none(metadata.get("max_tokens"))
        capabilities = metadata.get("capabilities") if isinstance(metadata.get("capabilities"), dict) else {}
        if _cap_supported(capabilities, "image_input"):
            _add_modality(card, "image")
        if _cap_supported(capabilities, "pdf_input"):
            _add_modality(card, "pdf")
            _add_modality(card, "file")
            card.supports_file_input = True
        if _cap_supported(capabilities, "structured_outputs"):
            card.supports_structured_output = True
        if _cap_supported(capabilities, "code_execution"):
            card.supports_code_execution = True
        if _cap_supported(capabilities, "thinking"):
            card.strengths.append("reasoning")
    elif provider == "google":
        card.context_window = card.context_window or _int_or_none(metadata.get("input_token_limit"))
        card.max_output_tokens = card.max_output_tokens or _int_or_none(metadata.get("output_token_limit"))
        actions = set(metadata.get("supported_actions") or [])
        if "embedContent" in actions:
            card.model_kind = "embedding"
            card.supports_embeddings = True
            card.supports_streaming = False
            card.modalities_output = ["embedding"]
            card.strengths.extend(["embeddings", "rag", "semantic_search"])
        if "generateContent" in actions:
            card.supports_streaming = True
        if metadata.get("thinking"):
            card.strengths.append("reasoning")


def _apply_metadata_pricing(card: CapabilityCard, metadata: dict[str, Any]) -> None:
    pricing = metadata.get("pricing")
    if not isinstance(pricing, dict):
        direct_keys = {
            "input_per_million_usd",
            "output_per_million_usd",
            "cached_input_per_million_usd",
            "input_usd_per_million",
            "output_usd_per_million",
        }
        pricing = {key: metadata[key] for key in direct_keys if key in metadata}
    if not pricing:
        return
    card.pricing = {**card.pricing, **pricing, "source": str(pricing.get("source") or "provider_metadata")}


def _apply_official_override(card: CapabilityCard) -> None:
    override = OFFICIAL_MODEL_OVERRIDES.get(card.model_ref.key)
    if override is None:
        return
    for field in ("context_window", "max_output_tokens", "cost_tier", "latency_tier", "quality_tier"):
        value = override.get(field)
        if value is not None:
            setattr(card, field, value)
    if override.get("modalities_input"):
        card.modalities_input = list(override["modalities_input"])
    if override.get("modalities_output"):
        card.modalities_output = list(override["modalities_output"])
    if override.get("pricing"):
        card.pricing = {**card.pricing, **override["pricing"]}
    if override.get("strengths"):
        card.strengths.extend(override["strengths"])
    if "pdf" in card.modalities_input or "file" in card.modalities_input:
        card.supports_file_input = True
    if "image" in card.modalities_input:
        card.supports_file_input = card.supports_file_input or card.model_ref.provider in {"google", "anthropic"}


def _apply_family_inference(card: CapabilityCard) -> None:
    model = card.model_ref.model.lower()
    provider = card.model_ref.provider
    if card.model_kind == "embedding":
        return
    if provider == "openai":
        if "embedding" in model:
            card.model_kind = "embedding"
            card.modalities_output = ["embedding"]
            card.supports_embeddings = True
            card.supports_streaming = False
            card.strengths.extend(["embeddings", "rag", "semantic_search"])
            return
        if model.startswith(("gpt-image", "chatgpt-image")):
            card.model_kind = "image"
            card.modalities_input = ["text", "image"]
            card.modalities_output = ["image"]
            card.supports_tools = False
            card.supports_structured_output = False
            card.strengths.extend(["image_generation", "vision"])
            return
        if model.startswith(("gpt-audio", "gpt-realtime")):
            card.model_kind = "audio" if model.startswith("gpt-audio") else "realtime"
            card.modalities_input = ["text", "audio"]
            card.modalities_output = ["text", "audio"]
            card.strengths.extend(["audio", "low_latency"])
            return
        if model.startswith(("whisper", "gpt-4o-transcribe", "gpt-4o-mini-transcribe")):
            card.model_kind = "transcription"
            card.modalities_input = ["audio"]
            card.modalities_output = ["text"]
            card.supports_tools = False
            card.supports_structured_output = False
            card.strengths.append("transcription")
            return
        if model.startswith(("tts-", "gpt-4o-mini-tts")):
            card.model_kind = "tts"
            card.modalities_input = ["text"]
            card.modalities_output = ["audio"]
            card.supports_tools = False
            card.supports_structured_output = False
            card.strengths.append("audio")
            return
        if model.startswith("sora"):
            card.model_kind = "video"
            card.modalities_input = ["text", "image"]
            card.modalities_output = ["video"]
            card.supports_tools = False
            card.supports_structured_output = False
            card.strengths.append("video_generation")
            return
        if model.startswith("gpt-") or model.startswith("chat"):
            card.supports_tools = True
            card.supports_structured_output = True
            _add_modality(card, "image")
            card.strengths.extend(["tool_use", "structured_output"])
        if "mini" in model or "nano" in model:
            card.strengths.extend(["low_cost", "low_latency"])
        if "codex" in model:
            card.strengths.extend(["coding", "agentic", "tool_use"])
    elif provider == "anthropic":
        if model.startswith("claude"):
            card.supports_tools = True
            card.supports_structured_output = True
            card.supports_file_input = True
            _add_modality(card, "image")
            _add_modality(card, "pdf")
            card.strengths.extend(["reasoning", "coding", "agentic", "critique", "tool_use"])
    elif provider == "google":
        if model.startswith("gemini") and "embedding" not in model:
            card.supports_tools = True
            card.supports_structured_output = True
            card.supports_file_input = True
            card.supports_web = True
            _add_modality(card, "image")
            _add_modality(card, "pdf")
            if any(name in model for name in ["flash", "pro", "live"]):
                _add_modality(card, "audio")
                _add_modality(card, "video")
            card.strengths.extend(["multimodal", "structured_output", "tool_use"])
        if model.startswith(("imagen", "nano-banana")) or "image" in model:
            card.model_kind = "image"
            card.modalities_input = ["text", "image"]
            card.modalities_output = ["text", "image"]
            card.supports_tools = False
            card.strengths.extend(["image_generation", "vision"])
        if model.startswith("veo"):
            card.model_kind = "video"
            card.modalities_input = ["text", "image"]
            card.modalities_output = ["video"]
            card.supports_tools = False
            card.supports_structured_output = False
            card.strengths.append("video_generation")
        if model.startswith("lyria"):
            card.model_kind = "music"
            card.modalities_input = ["text"]
            card.modalities_output = ["audio"]
            card.supports_tools = False
            card.supports_structured_output = False
            card.strengths.append("audio_generation")
    elif provider == "ollama":
        card.strengths.extend(["privacy"])
        if any(name in model for name in ["coder", "code", "devstral"]):
            card.strengths.extend(["coding", "agentic"])
        if any(name in model for name in ["glm", "kimi", "minimax", "deepseek", "qwen3"]):
            card.strengths.extend(["reasoning", "agentic"])
        if any(name in model for name in ["vl", "vision", "llava", "moondream", "gemini-3-flash"]):
            _add_modality(card, "image")
            card.supports_file_input = True
            card.strengths.append("multimodal")
        if "gpt-oss" in model:
            card.supports_tools = True
            card.supports_structured_output = True
            card.strengths.extend(["reasoning", "tool_use", "structured_output"])


def _apply_lifecycle_profile(card: CapabilityCard) -> None:
    decision = _model_decision(card)
    _apply_curated_defaults(card)
    specialized = SPECIALIZED_MODEL_USES.get(card.model_ref.key)
    if specialized:
        card.model_kind = specialized.get("model_kind", card.model_kind)
        if specialized.get("modalities_input"):
            card.modalities_input = list(specialized["modalities_input"])
        if specialized.get("modalities_output"):
            card.modalities_output = list(specialized["modalities_output"])
        card.strengths.extend(specialized.get("preferred_when", []))

    notice = DEPRECATED_MODEL_NOTICES.get(card.model_ref.key)
    if notice:
        source_key = str(notice.get("source", ""))
        card.deprecation = {
            "status": notice.get("lifecycle", "deprecated"),
            "shutdown_date": notice.get("shutdown_date"),
            "replacement": notice.get("replacement"),
            "source": SOURCE_URLS.get(source_key, source_key),
            "noted_at": SOURCE_SNAPSHOT_DATE,
        }

    profile = dict(card.natural_profile or {})
    profile.update(
        {
            "routing_status": decision["routing_status"],
            "lifecycle": decision["lifecycle"],
            "production_default": decision["production_default"],
            "requires_opt_in": decision["requires_opt_in"],
            "status_reason": decision["status_reason"],
        }
    )
    if decision.get("replacement"):
        profile["replacement"] = decision["replacement"]
    card.natural_profile = profile

    hints = dict(card.routing_hints or {})
    preferred = sorted(set(hints.get("preferred_when", []) + decision.get("preferred_when", [])))
    avoid = sorted(set(hints.get("avoid_when", []) + decision.get("avoid_when", [])))
    hints.update(
        {
            "routing_status": decision["routing_status"],
            "lifecycle": decision["lifecycle"],
            "production_default": decision["production_default"],
            "requires_opt_in": decision["requires_opt_in"],
            "preferred_when": preferred,
            "avoid_when": avoid,
        }
    )
    card.routing_hints = hints

    decision_bench = dict(card.benchmarks.get("decision_profile", {}))
    decision_bench.update(
        {
            "snapshot_date": SOURCE_SNAPSHOT_DATE,
            "routing_status": decision["routing_status"],
            "lifecycle": decision["lifecycle"],
            "source": "crupier_curated_profile",
        }
    )
    card.benchmarks["decision_profile"] = decision_bench


def _apply_curated_defaults(card: CapabilityCard) -> None:
    defaults = CURATED_MODEL_DEFAULTS.get(card.model_ref.key)
    if not defaults:
        return
    for field in ("quality_tier", "cost_tier", "latency_tier"):
        if getattr(card, field) == "unknown" and defaults.get(field):
            setattr(card, field, defaults[field])
    if defaults.get("strengths"):
        card.strengths.extend(defaults["strengths"])
    if card.pricing.get("confidence") in {None, "unknown"} and card.model_ref.provider == "ollama":
        card.pricing = {
            **card.pricing,
            "confidence": "account_dependent",
            "source": SOURCE_URLS["ollama_cloud"],
        }


def _model_decision(card: CapabilityCard) -> dict[str, Any]:
    key = card.model_ref.key
    model = card.model_ref.model.lower()
    provider = card.model_ref.provider
    specialized = SPECIALIZED_MODEL_USES.get(key)

    if key in DEPRECATED_MODEL_NOTICES:
        notice = DEPRECATED_MODEL_NOTICES[key]
        lifecycle = str(notice.get("lifecycle", "deprecated"))
        replacement = notice.get("replacement")
        return {
            "routing_status": lifecycle,
            "lifecycle": lifecycle,
            "production_default": False,
            "requires_opt_in": True,
            "replacement": replacement,
            "preferred_when": [],
            "avoid_when": ["production_default"],
            "status_reason": _deprecation_reason(card, notice),
        }

    if specialized or card.model_kind != "chat":
        use = specialized or {}
        return {
            "routing_status": "specialized",
            "lifecycle": _base_lifecycle(card),
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": list(use.get("preferred_when", [])),
            "avoid_when": ["general_chat", "default_routing"],
            "status_reason": use.get("status_reason")
            or f"{card.model_ref.key} is specialized for {card.model_kind} workloads.",
        }

    if key in PRODUCTION_RECOMMENDED_MODELS:
        return {
            "routing_status": "recommended",
            "lifecycle": _base_lifecycle(card),
            "production_default": True,
            "requires_opt_in": False,
            "preferred_when": [],
            "avoid_when": [],
            "status_reason": "Curated Crupier production-routing model as of 2026-06-20.",
        }

    if key in OPT_IN_RECOMMENDED_MODELS:
        return {
            "routing_status": "opt_in",
            "lifecycle": _base_lifecycle(card),
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": [],
            "avoid_when": ["strict_production_default"],
            "status_reason": "Useful current model, but preview/managed/special routing should be explicit.",
        }

    if key in EXPENSIVE_OPT_IN_MODELS:
        return {
            "routing_status": "opt_in",
            "lifecycle": _base_lifecycle(card),
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": [],
            "avoid_when": ["default_routing", "low_cost"],
            "status_reason": "Capable but excluded from Crupier defaults because cost can be high; add it explicitly if needed.",
        }

    if card.model_ref.stability in {"preview", "experimental"}:
        return {
            "routing_status": "opt_in",
            "lifecycle": card.model_ref.stability,
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": [],
            "avoid_when": ["strict_production_default"],
            "status_reason": f"{card.model_ref.stability.title()} models require explicit opt-in.",
        }

    if card.model_ref.stability == "latest" or model.endswith("-latest") or "latest" in model:
        return {
            "routing_status": "opt_in",
            "lifecycle": "latest_alias",
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": [],
            "avoid_when": ["reproducible_production"],
            "status_reason": "Latest aliases can move under the same model ID; pin stable IDs for production routing.",
        }

    if _looks_like_snapshot(model):
        return {
            "routing_status": "opt_in",
            "lifecycle": "snapshot",
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": [],
            "avoid_when": ["default_routing"],
            "status_reason": "Date-pinned snapshots are reproducible but not the curated default model family.",
        }

    if provider == "openai" and "codex" in model:
        return {
            "routing_status": "opt_in",
            "lifecycle": _base_lifecycle(card),
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": ["coding", "agentic", "console"],
            "avoid_when": ["default_routing", "non_coding_routes"],
            "status_reason": "Codex-family models are coding-agent candidates, but must be enabled explicitly for each project.",
        }

    if provider == "openai" and "deep-research" in model:
        return {
            "routing_status": "specialized",
            "lifecycle": _base_lifecycle(card),
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": ["research", "long_context"],
            "avoid_when": ["default_routing", "low_latency"],
            "status_reason": "Deep-research models are specialized research routes, not general default chat models.",
        }

    if provider == "openai" and "search-api" in model:
        return {
            "routing_status": "specialized",
            "lifecycle": _base_lifecycle(card),
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": ["research", "search"],
            "avoid_when": ["default_routing"],
            "status_reason": "Search API models are specialized retrieval/search routes, not general default chat models.",
        }

    if provider == "openai" and (
        model.startswith(("gpt-3.5", "gpt-4", "gpt-4o", "gpt-4.1", "o1", "davinci", "babbage"))
        or model == "text-embedding-ada-002"
    ):
        return {
            "routing_status": "legacy",
            "lifecycle": "legacy",
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": [],
            "avoid_when": ["default_routing"],
            "status_reason": "Older OpenAI family kept visible for compatibility, not selected by Crupier defaults.",
        }

    if provider == "google" and model.startswith(("gemma", "aqa", "robotics")):
        return {
            "routing_status": "specialized",
            "lifecycle": _base_lifecycle(card),
            "production_default": False,
            "requires_opt_in": True,
            "preferred_when": [],
            "avoid_when": ["default_routing"],
            "status_reason": "Specialized Google model family; route only explicit matching workloads.",
        }

    return {
        "routing_status": "unknown",
        "lifecycle": _base_lifecycle(card),
        "production_default": False,
        "requires_opt_in": True,
        "preferred_when": [],
        "avoid_when": ["default_routing"],
        "status_reason": "Discovered model without enough curated Crupier evidence for default routing.",
    }


def _base_lifecycle(card: CapabilityCard) -> str:
    if card.model_ref.stability in {"preview", "experimental"}:
        return card.model_ref.stability
    if card.model_ref.stability == "latest":
        return "latest_alias"
    if _looks_like_snapshot(card.model_ref.model.lower()):
        return "snapshot"
    return "current"


def _looks_like_snapshot(model: str) -> bool:
    return any(f"-{year}" in model for year in range(2024, 2027)) or any(
        model.endswith(str(year)) for year in range(2024, 2027)
    )


def _deprecation_reason(card: CapabilityCard, notice: dict[str, Any]) -> str:
    status = str(notice.get("lifecycle", "deprecated"))
    shutdown = notice.get("shutdown_date")
    replacement = notice.get("replacement")
    pieces = [f"{card.model_ref.key} is marked {status} in provider documentation"]
    if shutdown:
        pieces.append(f"shutdown date {shutdown}")
    if replacement:
        pieces.append(f"recommended replacement {replacement}")
    return "; ".join(pieces) + "."


def _apply_capability_status(card: CapabilityCard) -> None:
    status = card.capability_status
    if _status_is_failed(status.get("tool_call")):
        card.supports_tools = False
        _remove_strength(card, "tool_use")
    if _status_is_failed(status.get("structured_output")):
        card.supports_structured_output = False
        _remove_strength(card, "structured_output")
    if _status_is_failed(status.get("streaming")):
        card.supports_streaming = False
    if _status_is_failed(status.get("embeddings")):
        card.supports_embeddings = False
        _remove_strength(card, "embeddings")
        _remove_strength(card, "rag")
        _remove_strength(card, "semantic_search")
    if _status_is_supported(status.get("tool_call")):
        card.supports_tools = True
        card.strengths.append("tool_use")
    if _status_is_supported(status.get("structured_output")):
        card.supports_structured_output = True
        card.strengths.append("structured_output")
    if _status_is_supported(status.get("streaming")):
        card.supports_streaming = True
    if _status_is_supported(status.get("embeddings")):
        card.model_kind = "embedding"
        card.supports_embeddings = True
        card.strengths.extend(["embeddings", "rag", "semantic_search"])


def _merged_skill_scores(card: CapabilityCard) -> dict[str, Any]:
    scores: dict[str, float] = {}
    for strength in card.strengths:
        scores[_score_key(strength)] = max(scores.get(_score_key(strength), 0.0), 7.0)
    if card.quality_tier == "frontier":
        scores["reasoning"] = max(scores.get("reasoning", 0.0), 8.2)
    elif card.quality_tier == "strong":
        scores["reasoning"] = max(scores.get("reasoning", 0.0), 7.0)
    if card.cost_tier == "low":
        scores["low_cost"] = max(scores.get("low_cost", 0.0), 8.0)
    if card.latency_tier == "fast":
        scores["low_latency"] = max(scores.get("low_latency", 0.0), 8.0)
    if "image" in card.modalities_input:
        scores["vision"] = max(scores.get("vision", 0.0), 7.0)
    if "pdf" in card.modalities_input:
        scores["pdf"] = max(scores.get("pdf", 0.0), 7.5)
        scores["document_extraction"] = max(scores.get("document_extraction", 0.0), 7.5)
    if card.supports_tools:
        scores["tool_use"] = max(scores.get("tool_use", 0.0), 7.5)
    if card.supports_structured_output:
        scores["structured_output"] = max(scores.get("structured_output", 0.0), 7.5)
    if card.supports_embeddings:
        scores["embeddings"] = max(scores.get("embeddings", 0.0), 9.0)
        scores["rag"] = max(scores.get("rag", 0.0), 9.0)
    override = OFFICIAL_MODEL_OVERRIDES.get(card.model_ref.key)
    if override:
        for skill, value in override.get("skill_scores", {}).items():
            scores[skill] = max(scores.get(skill, 0.0), float(value))
    curated = CURATED_MODEL_DEFAULTS.get(card.model_ref.key)
    if curated:
        for skill, value in curated.get("skill_scores", {}).items():
            scores[skill] = max(scores.get(skill, 0.0), float(value))
    model = card.model_ref.model.lower()
    if card.model_ref.provider == "ollama":
        if "glm-5.2" in model:
            scores.update(_max_scores(scores, {"reasoning": 8.6, "agentic": 8.6, "coding": 8.2, "orchestration": 8.6}))
        if "qwen3-coder" in model or "kimi-k2.7-code" in model:
            scores.update(_max_scores(scores, {"coding": 8.8, "agentic": 8.4, "tool_use": 7.5}))
        if "minimax-m3" in model:
            scores.update(_max_scores(scores, {"agentic": 8.4, "coding": 8.2, "multimodal": 8.0}))
    for key, value in card.skill_scores.items():
        if isinstance(value, int | float):
            scores[key] = max(scores.get(key, 0.0), float(value))
    return {key: round(value, 2) for key, value in sorted(scores.items()) if value > 0}


def _natural_profile(card: CapabilityCard) -> dict[str, Any]:
    existing = dict(card.natural_profile or {})
    override = OFFICIAL_MODEL_OVERRIDES.get(card.model_ref.key) or {}
    curated = CURATED_MODEL_DEFAULTS.get(card.model_ref.key) or {}
    best = _top_skill_labels(card.skill_scores, minimum=8.0, limit=6)
    if not best:
        best = _top_skill_labels(card.skill_scores, minimum=7.0, limit=4)
    avoid: list[str] = []
    if card.cost_tier == "high":
        avoid.append("high-volume routes where cost matters more than quality")
    if card.latency_tier in {"medium", "slow"}:
        avoid.append("ultra-low-latency interactions unless quality justifies the wait")
    if card.model_ref.stability in {"preview", "experimental"}:
        avoid.append("strict production routes unless preview models are explicitly allowed")
    if card.model_kind == "embedding":
        avoid.append("chat generation or tool-use routes")
    summary = override.get("natural_summary") or curated.get("natural_summary") or existing.get("summary") or _default_summary(card, best)
    profile = {
        "summary": summary,
        "routing_status": existing.get("routing_status"),
        "lifecycle": existing.get("lifecycle"),
        "production_default": existing.get("production_default"),
        "requires_opt_in": existing.get("requires_opt_in"),
        "status_reason": existing.get("status_reason"),
        "replacement": existing.get("replacement"),
        "best_for": existing.get("best_for") or best,
        "avoid_for": existing.get("avoid_for") or avoid,
        "tradeoffs": existing.get("tradeoffs") or _tradeoffs(card),
        "confidence": existing.get("confidence") or _profile_confidence(card),
        "generated_from": sorted(set(existing.get("generated_from", []) + ["capability_card", "provider_discovery", "crupier_rules"])),
        "updated_at": existing.get("updated_at") or SOURCE_SNAPSHOT_DATE,
    }
    return {key: value for key, value in profile.items() if value not in (None, [], {})}


def _routing_hints(card: CapabilityCard) -> dict[str, Any]:
    existing = dict(card.routing_hints or {})
    preferred = [skill for skill, value in card.skill_scores.items() if isinstance(value, int | float) and value >= 8.0]
    avoid: list[str] = []
    if card.model_kind == "embedding":
        avoid.extend(["chat", "tool_use"])
    if card.cost_tier == "high":
        avoid.append("low_cost")
    if card.latency_tier in {"medium", "slow"}:
        avoid.append("low_latency")
    strategy_bias: list[str] = []
    if "critique" in preferred:
        strategy_bias.append("critique_repair")
    if {"research", "reasoning"}.intersection(preferred):
        strategy_bias.append("fusion")
    if {"low_cost", "low_latency", "orchestration"}.intersection(preferred):
        strategy_bias.append("single")
        strategy_bias.append("cascade")
    if card.model_kind == "embedding":
        strategy_bias = ["embedding"]
    hints = {
        "routing_status": existing.get("routing_status"),
        "lifecycle": existing.get("lifecycle"),
        "production_default": existing.get("production_default", False),
        "requires_opt_in": existing.get("requires_opt_in", True),
        "preferred_when": sorted(set(existing.get("preferred_when", []) + preferred)),
        "avoid_when": sorted(set(existing.get("avoid_when", []) + avoid)),
        "strategy_bias": sorted(set(existing.get("strategy_bias", []) + strategy_bias)),
        "orchestrator_candidate": bool(
            existing.get("orchestrator_candidate")
            or card.skill_scores.get("orchestration", 0) >= 8
            or (card.supports_structured_output and card.supports_tools and card.latency_tier == "fast")
        ),
        "confidence": existing.get("confidence") or _profile_confidence(card),
    }
    return {key: value for key, value in hints.items() if value not in (None, [], {})}


def _evidence(card: CapabilityCard, *, provider_metadata: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(card.evidence or {})
    source_keys = set()
    override = OFFICIAL_MODEL_OVERRIDES.get(card.model_ref.key)
    if override:
        source_keys.update(override.get("sources", []))
    curated = CURATED_MODEL_DEFAULTS.get(card.model_ref.key)
    if curated:
        source_keys.update(curated.get("sources", []))
        source_keys.add("crupier_curated_profile")
    if provider_metadata:
        source_keys.add("provider_discovery")
    if card.deprecation:
        source = card.deprecation.get("source")
        for key, url in SOURCE_URLS.items():
            if source == url:
                source_keys.add(key)
                break
    if card.probe_results or card.capability_status:
        source_keys.add("crupier_probes")
    if card.local_eval_scores:
        source_keys.add("local_evals")
    source_keys.update(["arena_text", "artificial_analysis", "swe_bench", "livebench"])
    sources = list(evidence.get("sources", []))
    existing_labels = {item.get("label") for item in sources if isinstance(item, dict)}
    for key in sorted(source_keys):
        if key in existing_labels:
            continue
        if key == "provider_discovery":
            sources.append({"label": key, "confidence": "measured", "captured_at": SOURCE_SNAPSHOT_DATE})
        elif key == "crupier_curated_profile":
            sources.append({"label": key, "confidence": "curated", "captured_at": SOURCE_SNAPSHOT_DATE})
        elif key == "crupier_probes":
            sources.append({"label": key, "confidence": "verified", "captured_at": SOURCE_SNAPSHOT_DATE})
        elif key == "local_evals":
            sources.append({"label": key, "confidence": "project_evidence", "captured_at": SOURCE_SNAPSHOT_DATE})
        else:
            sources.append({"label": key, "url": SOURCE_URLS.get(key), "confidence": "official" if key.startswith(("openai", "anthropic", "google", "ollama")) else "external"})
    evidence.update(
        {
            "source_snapshot_date": evidence.get("source_snapshot_date", SOURCE_SNAPSHOT_DATE),
            "claim_policy": "verified > measured > official > external > inferred > unknown",
            "sources": sources,
        }
    )
    return evidence


def _default_summary(card: CapabilityCard, best: list[str]) -> str:
    if card.model_kind == "embedding":
        return f"{card.model_ref.key} is an embedding model for retrieval and semantic search routes."
    if best:
        return f"{card.model_ref.key} is best suited for " + ", ".join(best[:3]) + "."
    return f"{card.model_ref.key} has limited decision metadata; run discovery, probes, and evals before production routing."


def _tradeoffs(card: CapabilityCard) -> list[str]:
    tradeoffs: list[str] = []
    if card.cost_tier != "unknown":
        tradeoffs.append(f"cost tier is {card.cost_tier}")
    if card.latency_tier != "unknown":
        tradeoffs.append(f"latency tier is {card.latency_tier}")
    if card.quality_tier != "unknown":
        tradeoffs.append(f"quality tier is {card.quality_tier}")
    if card.model_ref.provider == "ollama":
        tradeoffs.append("Ollama Cloud availability and quotas depend on the configured account")
    return tradeoffs


def _profile_confidence(card: CapabilityCard) -> str:
    if card.probe_results or any(_status_is_supported(value) for value in card.capability_status.values()):
        return "high"
    if OFFICIAL_MODEL_OVERRIDES.get(card.model_ref.key) or card.context_window or card.max_output_tokens:
        return "medium"
    if card.pricing.get("source") == "provider_discovery" or card.model_ref.source == "discovered":
        return "medium"
    return "low"


def _top_skill_labels(scores: dict[str, Any], *, minimum: float, limit: int) -> list[str]:
    ranked = sorted(
        (
            (skill, float(value))
            for skill, value in scores.items()
            if isinstance(value, int | float) and float(value) >= minimum
        ),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    return [HUMAN_SKILL_LABELS.get(skill, skill.replace("_", " ")) for skill, _ in ranked[:limit]]


def _max_scores(existing: dict[str, float], updates: dict[str, float]) -> dict[str, float]:
    return {key: max(existing.get(key, 0.0), value) for key, value in updates.items()}


def _score_key(strength: str) -> str:
    aliases = {
        "local": "privacy",
        "schema_validity": "structured_output",
        "semantic_search": "embeddings",
        "quality": "reasoning",
    }
    return aliases.get(strength, strength)


def _cap_supported(capabilities: dict[str, Any], key: str) -> bool:
    value = capabilities.get(key)
    return isinstance(value, dict) and value.get("supported") is True


def _status_is_supported(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") in {"verified", "inferred"}


def _status_is_failed(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == "failed"


def _remove_strength(card: CapabilityCard, strength: str) -> None:
    card.strengths = [item for item in card.strengths if item != strength]


def _add_modality(card: CapabilityCard, modality: str) -> None:
    if modality not in card.modalities_input:
        card.modalities_input.append(modality)


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _modality_sort_key(value: str) -> tuple[int, str]:
    order = {"text": 0, "image": 1, "pdf": 2, "file": 3, "audio": 4, "video": 5, "embedding": 6}
    return (order.get(value, 99), value)
