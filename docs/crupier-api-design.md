# Crupier: Diseno de API Publica

Fecha: 2026-06-18  
Estado: diseno de API con implementacion inicial `0.1.0` en `src/crupier`, incluyendo adapters reales de texto para OpenAI, Anthropic Claude y Ollama Cloud.

## Objetivo

Definir como se usaria Crupier desde Python, CLI y configuracion de proyecto. La API debe servir primero a agentes, pero sin cerrarse a RAG, chat apps, jobs batch, notebooks, herramientas internas o backends web.

El diseno busca cuatro cosas:

1. Primer uso simple.
2. Control avanzado cuando haga falta.
3. Trazabilidad suficiente para confiar en decisiones dinamicas.
4. Adopcion progresiva en proyectos existentes: proxy compatible, autopatch, cliente compatible o SDK nativo.

## Forma Recomendada

La API principal es propia:

```python
from crupier import Crupier

crupier = Crupier.from_project()

result = crupier.deal(
    task="Review this agent plan and choose the best model route.",
    input=agent_plan,
    mode="agentic",
)

print(result.output_text)
```

`deal` es la metafora central: Crupier reparte la mano de modelos.

## Modos de Adopcion

Crupier debe poder entrar en proyectos existentes con el menor cambio posible:

- Cero cambios de codigo cuando el proyecto acepta cambiar `OPENAI_BASE_URL` hacia un server compatible.
- Una linea cuando se use autopatch: `import crupier; crupier.install()`.
- Cambio de import cuando se use cliente compatible: `from crupier.compat.openai import OpenAI`.
- Integracion completa cuando se use SDK nativo: `Crupier.from_project().deal(...)`.

El SDK nativo sigue siendo el contrato mas expresivo, pero no debe ser la unica puerta de entrada.

Estado implementado:

```python
from crupier.compat.openai import OpenAI

client = OpenAI(project=".")
response = client.responses.create(
    model="gpt-5.4-mini",
    input="Resume esto",
)

print(response.output_text)
```

```python
import crupier
crupier.install("openai")
```

La compatibilidad inicial cubre `responses.create`, `chat.completions.create`, `embeddings.create`, objetos respuesta tipo atributo/dict, `model_dump()`, stream compatible con eventos Responses y chunks Chat Completions, errores HTTP OpenAI-like y extraccion de content parts de imagen/archivo hacia el planner multimodal.

Server local inicial:

```bash
crupier serve --port 8787
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
```

Endpoints implementados: `GET /health`, `GET /v1/models`, `POST /v1/responses`, `POST /v1/chat/completions` y `POST /v1/embeddings`. El server compatible devuelve `x-request-id`, JSON de error estilo OpenAI y SSE tipado para Responses.

## Principios de API

- Una entrada simple debe ser facil.
- Una entrada avanzada debe ser explicita.
- La configuracion vive en `crupier.toml`, pero puede sobreescribirse por codigo.
- Sync, async y streaming deben compartir semantica.
- Las trazas son opcionales para la app, pero siempre disponibles si los constraints lo permiten.
- Los errores deben explicar proveedor, modelo, ruta y sugerencia de arreglo.
- La API no debe obligar a usar OpenRouter ni ningun gateway externo.
- Los modelos permitidos los define el usuario.

## Instalacion Conceptual

Dependencia base:

```bash
pip install crupier
```

Extras por proveedor:

```bash
pip install "crupier[openai]"
pip install "crupier[anthropic]"
pip install "crupier[google]"
pip install "crupier[ollama]"
pip install "crupier[openrouter]"
pip install "crupier[all]"
```

Recomendacion: mantener dependencia base ligera. Cada adapter debe vivir en extra opcional.

## Inicio de Proyecto

```bash
crupier init
```

Genera:

```text
crupier.toml
.crupier/
  registry/
  profiles/
  evals/
  traces/
```

Despues:

```bash
crupier update
```

## Configuracion Minima

```toml
[project]
name = "my-agent"
default_profile = "agentic"

[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[models]
allow = ["openai:gpt-5.5"]

[logging]
mode = "metadata"
store_prompts = false
store_responses = false
```

## Configuracion Agentic

```toml
[project]
name = "agent-platform"
default_profile = "agentic"

[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.anthropic]
enabled = true
env_key = "ANTHROPIC_API_KEY"

[providers.google]
enabled = true
env_key = "GOOGLE_API_KEY"

[providers.ollama]
enabled = true
host = "https://ollama.com/api"
env_key = "OLLAMA_API_KEY"

[providers.openrouter]
enabled = false
mode = "byok"
env_key = "OPENROUTER_API_KEY"

[models]
allow = [
  "openai:gpt-5.5",
  "openai:gpt-5.4-mini",
  "anthropic:claude-opus-4-8",
  "google:gemini-3.5-flash",
  "ollama:gpt-oss:120b"
]

[routing]
default_strategy = "orchestrated"
allow_fusion = true
allow_parallel = true
allow_latest_aliases = false
max_cost_per_request_usd = 1.00
max_latency_ms = 30000
max_depth = 8
max_calls = 40

[orchestrator]
model = "openai:gpt-5.4-mini"
fallback_model = "google:gemini-3.5-flash"
temperature = 0
require_validated_plan = true

[logging]
mode = "metadata"
store_prompts = false
store_responses = false
redact_secrets = true
```

## Clase Principal

```python
from crupier import Crupier

crupier = Crupier.from_project()
```

Constructores:

```python
Crupier.from_project(path=".")
Crupier.from_config(config)
Crupier.from_toml("crupier.toml")
```

## Metodo Principal: `deal`

Firma conceptual:

```python
result = crupier.deal(
    task: str,
    input: object | None = None,
    *,
    mode: str | None = None,
    strategy: str | None = None,
    constraints: dict | None = None,
    tools: list | None = None,
    response_schema: object | None = None,
    metadata: dict | None = None,
    trace: bool | str = False,
)
```

Ejemplo simple:

```python
result = crupier.deal(
    task="Summarize this support thread",
    input=thread,
    mode="cheap",
)
```

Ejemplo con restricciones:

```python
result = crupier.deal(
    task="Plan the next actions for this customer escalation",
    input=ticket,
    mode="agentic",
    constraints={
        "max_cost_usd": 0.25,
        "max_latency_ms": 15000,
        "sensitive_data": "internal",
        "require_trace": "summary",
    },
)
```

## Async

```python
result = await crupier.adeal(
    task="Extract entities from these documents",
    input=documents,
    mode="structured",
)
```

## Streaming

Streaming debe existir, pero con semantica clara.

```python
for event in crupier.stream(
    task="Draft a migration plan",
    input=context,
    mode="research",
):
    if event.type == "text_delta":
        print(event.delta, end="")
    elif event.type == "route_selected":
        print(event.route.summary)
```

Eventos conceptuales:

- `route_started`
- `route_selected`
- `model_call_started`
- `model_call_completed`
- `text_delta`
- `tool_call_requested`
- `tool_call_completed`
- `fallback_triggered`
- `validation_failed`
- `final`
- `trace_available`

Para `fusion`, streaming debe poder tardar mas en empezar. Opcion recomendada:

- Stream de estado inmediato.
- Stream de salida final solo cuando el juez/escritor final empiezan.
- No mezclar respuestas parciales de panel con respuesta final salvo modo debug.

## Modos

`mode` expresa intencion de producto.

| Mode | Uso |
| --- | --- |
| `agentic` | Agentes, tool use, coding, autonomia, verificacion. |
| `quality` | Maxima calidad, coste menos importante. |
| `cheap` | Coste bajo con escalado si falla. |
| `fast` | Baja latencia. |
| `private` | Local-first, ZDR, no prompt logging. |
| `research` | Fusion, paneles, citas, contradicciones. |
| `structured` | JSON/schema, extraccion, validacion. |

## Estrategias

`strategy` expresa forma de ejecucion.

| Strategy | Descripcion |
| --- | --- |
| `orchestrated` | El orquestador elige estrategia. |
| `single` | Un modelo. |
| `fallback` | Lista priorizada. |
| `cascade` | Barato primero, escala si falla. |
| `panel` | Varios modelos, sin sintesis automatica. |
| `fusion` | Panel + juez + escritor final. |
| `critique_repair` | Generar, criticar, reparar. |
| `local_first` | Ollama configurado explicitamente como local/private antes de proveedores cerrados. |

## Resultado

```python
result.output_text
result.output_json
result.route
result.trace
result.cost
result.latency_ms
result.warnings
result.provider_metadata
```

### Ejemplo de `CrupierResult`

```json
{
  "output_text": "Use a cascade route for this task...",
  "route": {
    "strategy": "cascade",
    "models": ["openai:gpt-5.4-mini", "openai:gpt-5.5"],
    "summary": "Low-risk first pass with escalation on validation failure."
  },
  "cost": {
    "estimated_usd": 0.08,
    "actual_usd": 0.06
  },
  "latency_ms": 4200,
  "warnings": []
}
```

## Trazas

La app puede pedir trazas:

```python
result = crupier.deal(
    task="Choose a route",
    input=payload,
    mode="agentic",
    trace="summary",
)
```

Niveles:

- `False`: no devolver trace a la app.
- `summary`: decision humana corta.
- `debug`: detalle completo permitido por constraints.

Trace summary conceptual:

```json
{
  "trace_id": "trc_123",
  "chosen_strategy": "cascade",
  "chosen_models": ["openai:gpt-5.4-mini", "openai:gpt-5.5"],
  "excluded": [
    {
      "model": "anthropic:claude-opus-4-8",
      "reason": "Exceeded max_cost_usd for this route."
    }
  ],
  "policy_filters": ["max_cost", "stable_models_only"],
  "decision_reason": "Task is medium complexity and schema validation can catch failures."
}
```

## Structured Output

```python
from pydantic import BaseModel

class Extraction(BaseModel):
    company: str
    risk_level: str
    action_items: list[str]

result = crupier.deal(
    task="Extract customer risk from this ticket",
    input=ticket,
    mode="structured",
    response_schema=Extraction,
)

data = result.output_json
```

Reglas:

- Validar schema.
- Reparar si es razonable.
- No aceptar JSON valido pero semanticamente invalido.
- Exponer errores de validacion tipados.

## Tools

Crupier no debe inventar un sistema de tools incompatible. Debe aceptar tools en formato normalizado y mapear a cada proveedor cuando sea posible.

```python
result = crupier.deal(
    task="Plan and execute cost-aware repository checks",
    input=request,
    mode="agentic",
    tools=[read_file, run_tests],
    constraints={
        "allowed_tools": ["read_file", "run_tests"],
        "require_approval_for": ["write_file", "deploy"],
    },
)
```

Reglas:

- Tool allowlist por ruta.
- Idempotency keys para tools con efectos.
- Retries seguros por tool.
- Aprobacion humana para acciones sensibles.
- Trace de tool calls sin secretos.

## Agentes

Crupier debe poder recibir estado de agente:

```python
result = crupier.deal(
    task="Choose the next model route for this agent step",
    input=current_step,
    mode="agentic",
    metadata={
        "agent_id": "agent_123",
        "run_id": "run_456",
        "step_id": "step_003",
    },
    constraints={
        "max_depth": 8,
        "remaining_budget_usd": 0.75,
        "requires_reproducibility": True,
    },
)
```

Campos de agente recomendados:

- `agent_id`
- `run_id`
- `step_id`
- `parent_trace_id`
- `depth`
- `remaining_budget_usd`
- `remaining_calls`
- `goal_summary`
- `state_summary`
- `risk_accumulated`

## Fusion

Ejemplo:

```python
result = crupier.deal(
    task="Compare these three architecture options",
    input=options,
    mode="research",
    strategy="fusion",
    constraints={
        "max_panel_size": 3,
        "require_contradictions": True,
        "max_cost_usd": 2.00,
    },
    trace="summary",
)
```

Fusion debe devolver:

- respuesta final
- consenso
- contradicciones
- huecos
- riesgos
- modelos usados
- coste
- latencia

No debe exponer chain-of-thought privada de los modelos.

## OpenAI-like API Opcional

Para adopcion:

```python
result = crupier.responses.create(
    input="Summarize this",
    mode="cheap",
)
```

Esto debe mapear internamente a `deal`. No debe convertirse en la API canonica si limita la expresividad de Crupier.

## Model Management

### Listar modelos

```python
models = crupier.models.list()
```

### Descubrir modelos reales del proveedor

```python
models = crupier.models.discover(provider="openai")
models = crupier.models.discover(provider="anthropic")
models = crupier.models.discover(provider="google")
models = crupier.models.discover(provider="ollama")
```

CLI:

```bash
crupier models discover --provider openai
crupier models discover --provider anthropic
crupier models discover --provider google
crupier models discover --provider ollama
```

Las API keys deben venir de variables de entorno (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` o `GEMINI_API_KEY`, `OLLAMA_API_KEY`) o de un `.env` local ignorado por git, nunca de argumentos CLI ni de `crupier.toml`. Para Ollama Cloud, `OLLAMA_HOST=https://ollama.com/api` puede definir el host del adapter.

### Actualizar fichas con modelos vivos

```bash
crupier update --online
crupier update --online --provider openai
```

`update --online` consulta los proveedores habilitados y crea/actualiza capability cards para los modelos que existen y estan disponibles para esa cuenta en ese momento. No activa automaticamente todos esos modelos; la allowlist del proyecto se controla con `crupier models allow ...`.

Regla de producto:

- Desarrollo/exploracion: usar discovery online para ver el catalogo vivo.
- Produccion/auditoria: usar modelos explicitos y snapshots bloqueados para reproducibilidad.

### Snapshots de registry

```bash
crupier registry snapshot create baseline --allowed-only
crupier registry snapshot list
crupier registry snapshot diff baseline
crupier registry snapshot diff baseline current --json
crupier registry snapshot use baseline
crupier registry snapshot use baseline --restore-allowlist
```

Python:

```python
crupier.registry.snapshot_create("baseline", allowed_only=True)
crupier.registry.snapshot_diff("baseline", "current")
crupier.registry.snapshot_use("baseline", restore_allowlist=True)
```

### Seleccionar modelos permitidos

```bash
crupier models allow openai:gpt-5.5 anthropic:claude-opus-4-8 --replace
crupier models allow ollama:gpt-oss:120b
```

`claude:...` se acepta como alias y se normaliza a `anthropic:...`.

### Smoke tests reales

Despues de descubrir y seleccionar modelos permitidos, Crupier puede hacer una llamada real minima por proveedor/modelo:

```bash
crupier smoke --provider openai
crupier smoke --provider anthropic
crupier smoke --provider ollama
```

O modelos exactos:

```bash
crupier smoke --model openai:gpt-5.5
crupier smoke --model anthropic:claude-opus-4-8
crupier smoke --model ollama:gpt-oss:120b
```

El smoke usa un prompt inocuo, no guarda prompts/respuestas y no imprime salida del modelo salvo con `--show-output`.

### Capability probes

CLI:

```bash
crupier capabilities probe --provider openai
crupier capabilities probe --model openai:gpt-5.4-mini
crupier capabilities probe --provider openai --apply
crupier capabilities probe --probe structured_output --probe tool_call --probe streaming --apply
crupier capabilities probe --dry-run
crupier capabilities readiness
crupier capabilities readiness --strict
```

Python:

```python
report = crupier.capabilities.probe(
    ["openai:gpt-5.4-mini"],
    probes=["text_basic", "json_instruction", "structured_output", "tool_call"],
    apply=True,
)

readiness = crupier.capabilities.readiness(
    ["openai:gpt-5.4-mini"],
    strict=True,
)
```

`probe` ejecuta llamadas reales salvo con `--dry-run`. `--apply` persiste resultados en `capability_status` y `probe_results`; sin `--apply`, solo informa. Los probes disponibles son `text_basic`, `json_instruction`, `max_output_param`, `structured_output`, `tool_call` y `streaming`. Los probes nativos dependen de que el proveedor y el modelo soporten esa capacidad. No se guardan prompts ni respuestas crudas.

Para exigir evidencias verificadas en una ruta:

```python
result = crupier.deal(
    "Use this tool and return structured JSON.",
    tools=[...],
    response_schema={...},
    constraints={"require_verified_capabilities": True},
)
```

Sin ese constraint, Crupier puede usar soporte `inferred`, pero lo puntua por debajo de `verified` y bloquea capacidades con probe `failed`.

### Inspeccionar eleccion de modelo sin llamada real

```bash
crupier route "Compare two agent architectures and critique risks" --mode research
```

`route` devuelve estrategia, modelos elegidos y `selection_scores`, con terminos como calidad, preferencias del perfil, senales de la tarea, soporte de tools/structured output, evals locales y penalizaciones. Sirve para ajustar el algoritmo antes de gastar tokens o activar `--no-dry-run`.

### Ver capability card

```python
card = crupier.models.get("openai:gpt-5.5")
```

### Multimodalidad y archivos

La API acepta archivos sin convertir Crupier en gateway de seguridad:

```python
result = crupier.deal(
    task="Extrae los importes y riesgos de este contrato.",
    files=["contrato.pdf"],
    mode="structured",
)
```

El core debe convertir cada archivo en un `FileAsset` conceptual y decidir la representacion mas eficiente:

- imagen -> vision nativo u OCR/text extraction;
- PDF textual -> extraction + chunking;
- PDF escaneado -> OCR o vision por pagina;
- PDF con tablas -> table extraction + structured model;
- audio -> transcript salvo audio nativo verificado;
- video -> transcript + frames;
- CSV/XLSX/DOCX -> parser estructurado antes de LLM.

La decision se basa en capacidades verificadas, coste, latencia, calidad esperada y constraints declarados por el usuario. No hay inspeccion de contenido como producto de seguridad.

Estado implementado en `0.1.0`:

- `deal(..., files=[...])` acepta paths, URLs, dicts, bytes o `FileAsset`.
- `crupier route/deal --file ...` muestra `input_plan`.
- `FileRoutingPlan` omite URIs/rutas en `to_dict()` por defecto.
- Imagen -> `native_vision`, con filtro a modelos que declaren o verifiquen vision.
- PDF -> `extracted_text_chunks` por defecto para comparar modelos de texto por coste/latencia/calidad.
- `constraints={"require_native_file_input": True}` fuerza `native_pdf`/file input cuando aplique.
- La ejecucion real con archivos aun esta bloqueada hasta implementar mappings por adapter.

### Modelos Vectoriales

Crupier separa modelos conversacionales y modelos vectoriales mediante `CapabilityCard.model_kind`.

- Chat/multimodal: `model_kind="chat"`, salida `text`, posibles inputs `text/image/audio/video`.
- Embeddings: `model_kind="embedding"`, salida `embedding`, `supports_embeddings=true`, `embedding_dimensions` cuando se conoce o se verifica.

Ollama Cloud no se trata como "todos los modelos son vectoriales". Solo modelos dedicados como `embeddinggemma`, `all-minilm`, `nomic-embed-text`, `bge-*`, etc. se marcan como embedding por heuristica inicial, y la confirmacion real viene de `crupier capabilities probe --probe embeddings`.

Si `/v1/embeddings` recibe un modelo conocido como chat, la capa compatible debe fallar con error claro en vez de intentar usarlo como vectorial.

### Actualizar registry

```python
report = crupier.update(dry_run=True)
```

```python
report = crupier.update(apply=True)
```

`update` debe devolver diff:

```json
{
  "added_models": ["openai:gpt-5.4-nano"],
  "removed_models": ["openai:gpt-5.4-mini"],
  "modified_models": ["google:gemini-3.5-flash"],
  "unchanged_models": ["openai:gpt-5.5"],
  "changed_models": [
    "google:gemini-3.5-flash",
    "openai:gpt-5.4-mini",
    "openai:gpt-5.4-nano"
  ],
  "diff": {
    "added": ["openai:gpt-5.4-nano"],
    "removed": ["openai:gpt-5.4-mini"],
    "changed": [
      {
        "model": "google:gemini-3.5-flash",
        "fields": ["supports_tools", "quality_tier"]
      }
    ],
    "unchanged": 1
  },
  "model_states": [
    {
      "model": "openai:gpt-5.4-mini",
      "provider": "openai",
      "states": ["allowed", "stale"]
    }
  ],
  "deprecated_models": [],
  "price_changes": [],
  "profile_changes": [
    {
      "profile": "agentic",
      "old_recommendation": "openai:gpt-5.5",
      "new_recommendation": "anthropic:claude-opus-4-8",
      "reason": "Local eval score improved for long-horizon tool use."
    }
  ],
  "requires_confirmation": true
}
```

## Evals

```python
report = crupier.evals.run(
    dataset="agentic-routing-smoke",
    profiles=["agentic", "structured"],
)
```

Dataset conceptual:

```json
{
  "name": "agentic-routing-smoke",
  "cases": [
    {
      "id": "case_001",
      "task": "Plan a cost-aware file migration",
      "input": "...",
      "expected_properties": {
        "uses_tools": true,
        "requires_approval_for_writes": true,
        "max_cost_usd": 0.50
      }
    }
  ]
}
```

## Errores

Errores tipados recomendados:

- `CrupierConfigError`
- `CrupierPolicyError`
- `CrupierProviderAuthError`
- `CrupierProviderRateLimitError`
- `CrupierProviderUnavailableError`
- `CrupierModelUnsupportedError`
- `CrupierRouteValidationError`
- `CrupierBudgetExceededError`
- `CrupierToolApprovalRequired`
- `CrupierStructuredOutputError`
- `CrupierUpdateRequiresConfirmation`

Ejemplo:

```python
try:
    result = crupier.deal(task="...", input=payload)
except CrupierProviderAuthError as exc:
    print(exc.provider)
    print(exc.env_key)
    print(exc.hint)
```

## Semantica de Fallback

Fallback permitido:

- rate limit
- timeout
- provider unavailable
- model unavailable
- unsupported transient feature
- structured output repair exhausted, si constraints lo permiten

Fallback no permitido por defecto:

- refusal del proveedor que no sea claramente un problema de capacidad/disponibilidad
- constraint de privacidad declarado por el proyecto
- region prohibida
- presupuesto excedido
- herramienta no permitida
- modelo no ZDR cuando ZDR es requerido

## Semantica de Logging

Configuracion por codigo:

```python
result = crupier.deal(
    task="Debug this route",
    input=payload,
    constraints={
        "store_prompt": False,
        "store_response": False,
        "trace_level": "summary",
    },
)
```

Modos:

- `metadata`: default.
- `redacted`: guarda contenido redaccionado.
- `full`: guarda contenido completo con TTL obligatorio.
- `off`: sin persistencia local salvo runtime.

## Contratos de Reproducibilidad

Modo dinamico:

```python
result = crupier.deal(task="...", input=payload, mode="agentic")
```

Modo locked:

```python
result = crupier.deal(
    task="...",
    input=payload,
    constraints={
        "locked_registry_snapshot": "reg_2026_06_18",
        "locked_profile_snapshot": "prof_agentic_002",
    },
)
```

Uso: auditoria, produccion regulada, bugs reproducibles.

## Requisitos para Implementacion Futura

Antes de escribir codigo del paquete, cerrar:

1. Nombre final: `crupier`.
2. Version minima de Python.
3. Formato exacto de `crupier.toml`.
4. Nombres finales de `mode` y `strategy`.
5. Si `responses.create` entra en v1 o queda para compatibilidad posterior.
6. Formato exacto de `CapabilityCard`.
7. Formato exacto de `RoutePlan`.
8. Politica exacta de `crupier update`: online por defecto o `--online`.
9. Estructura de extras de instalacion.
10. Politica de trazas y TTL.
