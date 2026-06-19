# Crupier: Adopcion Drop-In

Fecha: 2026-06-18  
Estado: documento de producto y arquitectura.

Implementado en `0.1.0`:

- `crupier.compat.openai.OpenAI`
- `responses.create`
- `chat.completions.create`
- `embeddings.create`
- objetos respuesta con acceso por atributo/dict y `model_dump()`
- stream compatible con eventos Responses y chunks Chat Completions
- errores HTTP OpenAI-like con `x-request-id` en el server compatible
- extraccion de content parts de imagen/archivo hacia `FileRoutingPlan`
- `crupier.install("openai")` como autopatch opt-in
- `crupier serve` con `GET /health`, `GET /v1/models`, `POST /v1/responses`, `POST /v1/chat/completions` y `POST /v1/embeddings`

## Tesis

Crupier debe poder entrar en proyectos existentes sin obligar a reescribirlos. El caso ideal es:

> Me bajo un proyecto de GitHub, instalo Crupier, configuro claves/modelos, y el proyecto empieza a beneficiarse de routing, fallback, coste, latencia, multimodalidad y trazas con el minimo cambio posible.

Esto no significa que todos los proyectos puedan quedar mejorados con literalmente cero cambios. Depende de como llamen a modelos, que SDK usen, si fijan nombres de modelos en codigo, si usan streaming/tools/structured output, y si permiten cambiar `base_url` o cliente. El producto debe ofrecer varios niveles de adopcion.

## Niveles de Adopcion

### Nivel 0: Proxy Compatible

Objetivo: cero cambios de codigo cuando el proyecto ya usa un endpoint OpenAI-compatible o permite cambiar `base_url`.

Uso implementado inicial:

```bash
crupier serve --port 8787
export OPENAI_BASE_URL="http://localhost:8787/v1"
export OPENAI_API_KEY="<project-api-key>"
```

El proyecto sigue llamando a `client.chat.completions.create(...)` o `client.responses.create(...)`. Crupier recibe la llamada, interpreta modelo/tarea/params, decide ruta y llama a proveedores reales.

Ventajas:

- Adopcion casi cero cambios.
- Sirve para proyectos no Python.
- Centraliza budgets, trazas y routing.

Limitaciones:

- Solo cubre APIs compatibles con el contrato expuesto.
- Debe emular respuestas con mucha fidelidad.
- Streaming nativo/tools/files tienen que mapearse exactamente.

### Nivel 1: Monkeypatch/Autopatch Python

Objetivo: minimo cambio de codigo cuando el proyecto usa SDKs Python conocidos.

Uso esperado:

```python
import crupier
crupier.install("openai")
```

O via entorno:

```bash
CRUPIER_AUTOPATCH=openai,anthropic python app.py
```

Crupier intercepta llamadas de SDKs soportados:

- OpenAI Python SDK.
- Anthropic Python SDK.
- Google Gemini SDK.
- Ollama Python/client HTTP si aplica.
- LiteLLM/OpenRouter/OpenAI-compatible si aplica.

Ventajas:

- Permite proyectos Python existentes con muy pocos cambios.
- Mantiene gran parte del runtime dentro del proceso.
- Puede preservar objetos/respuestas del SDK original.

Limitaciones:

- Monkeypatching debe ser opt-in y reversible.
- Versiones de SDK cambian y pueden romper compatibilidad.
- Hay que cubrir sync, async, streaming y errores.

### Nivel 2: Cliente Compatible

Objetivo: cambio pequeño y explicito del cliente.

Uso esperado:

```python
from crupier.compat.openai import OpenAI

client = OpenAI()
```

El codigo del proyecto sigue pareciendo OpenAI-like, pero el cliente realmente llama a Crupier.

Ventajas:

- Menos fragil que monkeypatch.
- Buena experiencia para proyectos que aceptan cambiar imports.
- Facilita typed errors y trazas.

Limitaciones:

- Requiere tocar imports.
- Hay que emular bastante del SDK original.

### Nivel 3: SDK Nativo

Objetivo: maximo control para proyectos nuevos o integraciones profundas.

Uso esperado:

```python
from crupier import Crupier

result = Crupier.from_project().deal(
    task="...",
    input=payload,
    files=[...],
    tools=[...],
    mode="agentic",
    constraints={...},
)
```

Ventajas:

- Expone toda la potencia de Crupier.
- Mejor para agentes, evals, multimodalidad y trazas ricas.
- Contrato propio, no limitado por APIs de terceros.

Limitaciones:

- Requiere adaptar codigo.

## Contrato de Compatibilidad

Para funcionar en proyectos existentes, Crupier debe entender y preservar:

- `model`
- `messages` / `input`
- `system` / developer instructions si existen
- `temperature`, `top_p`, `max_tokens`, `max_output_tokens`
- `stream`
- `tools` / `functions` / `tool_choice`
- `response_format` / JSON schema / Pydantic si aplica
- archivos e imagenes
- embeddings
- moderation u otros endpoints que deban pasar-through
- errores esperados por el SDK original
- usage/tokens/cost metadata
- request ids

La regla: si Crupier no puede mejorar una llamada sin romper contrato, debe hacer pass-through o fallar con error claro segun configuracion.

## Motor de Decision Para Proyectos Existentes

Cuando Crupier intercepta una llamada heredada, debe reconstruir un `RequestEnvelope`:

- Tarea inferida desde mensajes/input.
- Modelo solicitado por el proyecto.
- Perfil inferido o configurado.
- Features requeridas: tools, schema, streaming, multimodal, embeddings.
- Constraints del proyecto: allowlist, budget, latency, region, privacidad declarada.
- Modo de compatibilidad: strict, balanced, aggressive.

Modos propuestos:

- `strict`: respeta el modelo pedido salvo error, budget o incompatibilidad.
- `balanced`: puede sustituir por modelo equivalente/mejor si mejora coste/latencia/calidad.
- `aggressive`: puede usar cascade/fusion/fallback si la tarea lo justifica.
- `locked`: reproduce rutas fijadas por snapshot.

## Configuracion Drop-In

Ejemplo:

```toml
[compat]
mode = "balanced"
strict_response_shape = true
pass_through_unknown_endpoints = true
preserve_requested_model_by_default = false

[compat.openai]
enabled = true
intercept = ["responses", "chat.completions", "embeddings"]

[compat.anthropic]
enabled = true
intercept = ["messages"]

[compat.routing]
fallback_on_rate_limit = true
fallback_on_provider_down = true
upgrade_for_tools = true
downgrade_for_cost = true
max_extra_calls = 2
```

## Preflight Recomendado

Antes de poner Crupier delante de una app existente, ejecutar:

```bash
crupier update --online --dry-run
crupier update --online
crupier capabilities readiness
crupier verify
crupier audit
crupier code comments --write-report
```

`crupier verify --provider anthropic --provider ollama` mantiene OpenAI como baseline por defecto. Esto permite comparar si Claude y Ollama estan listos sin perder la ruta de referencia OpenAI.

`crupier audit --real` ejecuta canaries reales con presupuestos pequenos: smoke de texto, structured output, tool loop local, archivo de texto local e imagen nativa cuando hay un modelo vision en la allowlist. Su objetivo no es solo decir que el codigo pasa, sino dejar preguntas de revision humana sobre estrategia, modelos, coste, multimodalidad y explicabilidad.

`crupier code comments` genera comentarios para otros programadores sobre call sites de OpenAI/Anthropic/Ollama/Google, modelos hard-codeados y posibles credenciales en codigo. No modifica fuentes por defecto.

## Edge Cases Clave

- El proyecto espera exactamente una clase/respuesta del SDK original.
- El proyecto hace `isinstance(...)` sobre objetos OpenAI/Anthropic.
- El proyecto usa streaming SSE y espera eventos exactos.
- El proyecto usa tool calls y depende de IDs concretos.
- El proyecto fija `model="gpt-4o"` y Crupier lo cambia sin que el prompt lo permita.
- El proyecto usa embeddings; no se puede routear a modelos chat.
- El proyecto usa batch APIs o uploads/files lifecycle del proveedor.
- El proyecto mezcla sync/async.
- El proyecto reintenta por su cuenta y Crupier tambien reintenta, duplicando coste.
- El proyecto espera errores del proveedor para activar fallback propio.
- El proyecto usa librerias encima de OpenAI como LangChain, LlamaIndex, LiteLLM o Vercel AI SDK.
- El proyecto no permite cambiar env vars.
- El proyecto usa claves directas en codigo.
- El proyecto depende de parametros no portables.

## Requisitos Para Ser "Drop-In Ready"

1. Server OpenAI-compatible minimo.
2. Compat client OpenAI-like en Python.
3. Autopatch opt-in para OpenAI Python SDK.
4. Emulacion fiel de response objects y errores principales.
5. Streaming compatible.
6. Tool calls compatibles.
7. Structured output compatible.
8. File/image/PDF mapping compatible.
9. Embeddings pass-through o routing dedicado.
10. Pass-through para endpoints desconocidos.
11. Trace sin romper respuesta original.
12. Budgets/circuit breakers para evitar doble retry y coste duplicado.
13. Test matrix contra SDKs reales y versiones soportadas.
14. Documentacion de compatibilidad por framework.
15. Auditoria de adopcion (`crupier audit`) con checks humanos, canaries reales y comentarios de codigo.

## Orden Recomendado

1. OpenAI-compatible Python client. Hecho inicial.
2. Routing `strict` y `balanced`. Hecho inicial.
3. Autopatch opt-in. Hecho inicial.
4. OpenAI-compatible server para `responses.create`, `chat.completions.create` y `embeddings.create`. Hecho inicial.
5. Streaming SSE basico. Hecho inicial.
6. Pass-through estricto para endpoints no soportados.
7. Streaming SSE completo.
8. Structured output.
9. Tools.
10. Files/multimodal execution.
11. Embeddings routing avanzado por proveedor/modelo.
12. Anthropic Messages compat.
13. Google/Ollama compat.
14. Integraciones LangChain/LlamaIndex/LiteLLM.

## Definicion de Exito

Crupier es drop-in ready cuando se puede tomar una app existente que usa OpenAI Python SDK, cambiar solo `OPENAI_BASE_URL` o una linea de import, y obtener:

- misma forma de respuesta;
- routing explicable;
- fallback real;
- presupuesto respetado;
- modelos vivos actualizados;
- trazas opt-in;
- sin guardar prompts/respuestas por defecto;
- sin romper streaming/tools/structured output cuando la app los usa.
