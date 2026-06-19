# Crupier: Documento de Producto y Discovery

Fecha: 2026-06-18  
Estado: discovery base del producto; la implementacion inicial vive como release publica `0.1.0`.

## Resumen

Crupier seria un paquete de PyPI para que cualquier aplicacion de IA pueda escoger, combinar o contrastar modelos de distintos proveedores segun la peticion, las restricciones del producto y el riesgo de equivocarse.

La idea central no es "un wrapper mas". Es un motor de decision: un crupier que reparte la mano entre modelos, familias y proveedores. Puede elegir un unico modelo, lanzar un panel paralelo, pedir critica, activar fallback, hacer sintesis con juez, o degradar a un modelo Ollama configurado por el proyecto cuando la tarea no justifica coste frontier.

Este documento no define un MVP. Define la base de un producto serio: versionable, observable, compatible con proveedores modernos, preparado para edge cases y con una estrategia clara para no quedarse obsoleto cuando cambien modelos, APIs, precios, condiciones de datos o capacidades.

## Fuentes Consultadas

Consulta realizada el 2026-06-18.

- [OpenRouter Fusion docs](https://openrouter.ai/docs/guides/features/plugins/fusion)
- [OpenRouter Fusion Router docs](https://openrouter.ai/docs/guides/routing/routers/fusion-router)
- [OpenRouter Fusion server tool docs](https://openrouter.ai/docs/guides/features/server-tools/fusion)
- [OpenRouter routing guide](https://openrouter.ai/blog/insights/model-routing/)
- [OpenRouter Fusion benchmark announcement](https://openrouter.ai/blog/announcements/fusion-beats-frontier/)
- [OpenAI API models](https://developers.openai.com/api/docs/models)
- [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)
- [Claude models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Claude Opus 4.8 notes](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8)
- [Claude migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide)
- [Anthropic API data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)
- [Gemini API models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash)
- [Gemini 3.1 Pro Preview](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)
- [Gemini API changelog](https://ai.google.dev/gemini-api/docs/changelog)
- [Gemini API terms](https://ai.google.dev/gemini-api/terms)
- [Ollama Cloud docs](https://docs.ollama.com/cloud)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Ollama Cloud launch note](https://ollama.com/blog/cloud-models)
- [LiteLLM routing docs](https://docs.litellm.ai/docs/routing)
- [RouteLLM GitHub](https://github.com/lm-sys/routellm)
- [Portkey fallbacks docs](https://docs.portkey.ai/docs/product/ai-gateway/fallbacks)

## Senal de Mercado

OpenRouter acaba de empujar fuerte el patron "fusion": varios modelos trabajan en paralelo, un juez produce analisis estructurado y el modelo final redacta la respuesta. Sus docs lo posicionan para investigacion, critica experta y tareas donde el coste de errar supera el coste de varias llamadas. Tambien documentan proteccion contra recursion y un coste aproximado lineal con el numero de modelos de panel.

La oportunidad para Crupier es distinta:

- Ser libreria PyPI embebible, no solo gateway externo.
- Permitir BYOK y llamadas directas a OpenAI, Anthropic, Google y Ollama Cloud.
- Mantener control local de constraints, trazas, privacidad, evaluaciones y decisiones.
- Servir tanto a apps normales como a agentes, pipelines batch, herramientas internas y flujos de investigacion.
- Ofrecer rutas reproducibles y explicables, no solo una caja negra que "elige el modelo".

## Estado Actual de Proveedores

Este mapa debe tratarse como fotografia del 2026-06-18, no como verdad permanente.

| Proveedor | Modelos/capacidades relevantes hoy | Implicacion para Crupier |
| --- | --- | --- |
| OpenAI | La documentacion presenta `gpt-5.5` como modelo flagship para razonamiento/coding complejo, con variantes `gpt-5.4`, `gpt-5.4-mini` y `gpt-5.4-nano` para coste/latencia. | Adapter nativo para Responses API, herramientas, multimodalidad, structured output y razonamiento configurable. |
| Anthropic | `claude-opus-4-8` es modelo Opus capaz para razonamiento, coding agentico y autonomia; `claude-fable-5` aparece en guia de migracion con requisitos de retencion de 30 dias y restricciones ZDR. | El router debe conocer compatibilidad de parametros, effort/thinking, refusals, ZDR, y no mandar datos sensibles a modelos no elegibles. |
| Google Gemini | `gemini-3.5-flash` esta estable y orientado a rendimiento frontier sostenido en tareas agenticas/coding; `gemini-3.1-pro-preview` sigue siendo relevante para razonamiento avanzado en preview. Google documenta aliases stable/preview/latest/experimental y deprecaciones activas. | Adapter nativo a Gemini API, cuidado con aliases `latest`, deprecaciones, search grounding, code execution, URL context y cambios de schema. |
| Ollama Cloud | Cloud permite ejecutar modelos grandes sin GPU local y exponer `https://ollama.com` como host remoto; docs listan acceso directo, modelos cloud y compatibilidad parcial con OpenAI Responses API. | Adapter native para Ollama Cloud, con host local explicito opcional para entornos privados o desarrollo. |

## Tesis de Producto

Crupier debe responder una pregunta: "Dada esta peticion y estas restricciones, cual es la mejor mano de modelos que puedo jugar?"

La decision puede ser:

- Un solo modelo.
- Un modelo barato primero, con escalado si falla.
- Un modelo Ollama configurado por el proyecto por privacidad, coste o control operativo.
- Un modelo frontier para alta autonomia.
- Un panel multi-modelo con juez.
- Un panel barato que intenta igualar calidad frontier.
- Una ruta de critica/reparacion.
- Una ruta con consenso, quorum o contradicciones.
- Un fallback por proveedor, modelo, coste, latencia, region o condiciones de datos.

## Principios

1. Constraints antes que inteligencia: nunca se debe elegir un modelo que viole privacidad, region, coste, modalidad, ZDR, licencia o compliance.
2. Transparencia por defecto: cada respuesta debe poder devolver un `decision_trace` entendible.
3. Modelos fijados por defecto: usar IDs estables para produccion; `latest` solo como opt-in consciente.
4. Logging configurable: por defecto registrar metadata operativa; guardar prompts/respuestas debe ser opt-in por proyecto/ruta para depuracion, evals o datasets, con redaccion y controles claros.
5. Evaluacion continua: las reglas del router deben poder probarse contra datasets propios.
6. Observabilidad real: coste, tokens, latencia, errores, fallback, recusaciones, retries y calidad estimada.
7. Composabilidad: usable desde apps sync/async, agentes, notebooks, CLIs y backends.
8. Sin dependencia obligatoria de un gateway: OpenRouter puede ser adapter opcional BYOK, no una ruta recomendada por defecto ni un servicio provisto por Crupier.

## Usuarios Objetivo

- Desarrolladores Python que ya usan OpenAI/Claude/Gemini y quieren cambiar a routing multi-modelo sin reescribir su app.
- Equipos de agentes que necesitan modelos distintos por paso: planificar, ejecutar, verificar, resumir, extraer, razonar.
- Empresas con restricciones de privacidad, ZDR, region, costes, BYOK y auditoria.
- Equipos de producto que quieren optimizar coste/calidad sin quedarse atados a un proveedor.
- Investigadores o builders que comparan modelos y quieren paneles reproducibles.
- Devs que quieren usar Ollama Cloud o un host Ollama propio como parte del mismo sistema.

## Capacidades de Producto

### 1. Modelo de Entrada Unificado

Un `RequestEnvelope` debe normalizar:

- Mensajes/instrucciones.
- Modalidades: texto, imagen, audio, video, PDF, archivos.
- Herramientas/function calling.
- Structured output.
- Streaming.
- Contexto y memoria.
- Metadatos de usuario/proyecto.
- Politicas de privacidad, coste, latencia y region.
- Preferencias: calidad, coste, velocidad, verificabilidad, creatividad, determinismo.

### 2. Registry de Modelos

Crupier necesita un registro versionado de modelos con:

- Proveedor y endpoint.
- Modelo estable, preview, latest o experimental.
- Context window y max output.
- Modalidades soportadas.
- Tool calling y structured output.
- Web/search/file/code/computer-use support.
- Precio estimado.
- Latencia historica.
- ZDR/data retention.
- Region/provider restrictions.
- Parametros soportados y rechazados.
- Lifecycle: activo, deprecated, shutting down, preview, experimental.

Este registry no deberia vivir solo hardcodeado. Debe soportar:

- Snapshots versionados incluidos en el paquete.
- Refresh opcional desde fuentes oficiales/APIs cuando existan.
- Overrides del usuario.
- Bloqueo de versiones para produccion.
- Alertas cuando un alias `latest` cambia.

### 3. El Crupier: Orquestador

El orquestador debe producir un `RoutePlan` antes de llamar modelos. La decision inicial es que Crupier tenga orquestador desde el inicio, pero no como un LLM libre que decide "por intuicion". Debe ser un orquestador model-powered y limitado por datos: constraints, registry de capacidades, benchmarks, historico local, presupuestos, latencia y restricciones del proyecto.

El usuario deberia poder declarar los modelos que quiere considerar, lanzar un `update`, y que Crupier actualice las fichas de capacidades del proyecto: modelos disponibles, parametros soportados, costes, deprecaciones, puntos fuertes, benchmarks conocidos, resultados de evals locales y rutas recomendadas.

Factores de decision:

- Intencion: chat, codigo, razonamiento, extraccion, clasificacion, vision, audio, tool-use, investigacion, verificacion.
- Riesgo de error: bajo, medio, alto, critico.
- Complejidad estimada: simple, multi-step, long-context, multimodal, tool-heavy.
- Sensibilidad declarada: publico, interno, secreto, regulado.
- Presupuesto: por request, por usuario, por tenant, por job.
- SLA: latencia maxima, streaming requerido, batch permitido.
- Tolerancia a variabilidad.
- Necesidad de citas/fuentes o evidencia.
- Necesidad de consenso multi-modelo.
- Restricciones declaradas por el usuario.

### 4. Modos de Routing

- `single`: elegir el mejor modelo unico.
- `fallback`: lista priorizada ante error, rate limit, timeout o refusal compatible.
- `cascade`: modelo barato primero; escalar si confidence baja o validacion falla.
- `panel`: N modelos en paralelo sin sintesis automatica.
- `fusion`: panel + juez + respuesta final.
- `critique_repair`: generador + critico + reparador.
- `quorum`: aceptar si K de N coinciden.
- `arbiter`: varios modelos proponen, juez elige.
- `local_first`: Ollama configurado explicitamente como local/private antes de proveedor cerrado.
- `privacy_first`: solo modelos compatibles con los constraints de privacidad.
- `cost_floor`: menor coste que cumpla calidad minima.
- `latency_race`: varias llamadas, gana la primera que supere umbral.
- `eval_locked`: rutas fijas validadas por benchmarks internos.

### 5. Fusion Propia

La version Crupier de fusion no debe limitarse a copiar OpenRouter. Deberia soportar:

- Panel configurable de 1 a N modelos.
- Juez configurable.
- Escritor final configurable.
- Consenso, contradicciones, huecos, riesgos y nivel de confianza.
- Trazas separadas para panel, juez y respuesta final.
- Timeouts por rama.
- Degradacion si falla parte del panel.
- Prevencion de recursion.
- Presupuesto maximo de llamadas/tokens.
- Politicas de privacidad por rama.
- Modo "sin web" y modo "con web/tools".
- Modo "red-team": un modelo intenta romper la respuesta.

Importante: no almacenar ni exponer chain-of-thought privada de proveedores. El analisis debe ser resumen estructurado seguro.

### 6. Evaluacion y Calidad

Sin evaluacion, el router se convierte en opinion. Crupier debe incluir:

- Harness de evals offline.
- Datasets por dominio del usuario.
- Comparacion A/B de rutas.
- Metricas: exactitud, coste, latencia, refusal, tool success, schema validity, user rating.
- Replay de requests anonimizadas.
- Regression tests para cambios de modelos.
- Golden answers y jueces multiples.
- Deteccion de drift cuando cambia un modelo o alias.

### 7. Observabilidad

Cada ejecucion deberia poder producir:

- `route_id`
- `chosen_strategy`
- `candidate_models`
- `policy_filters_applied`
- `calls_made`
- `fallbacks_triggered`
- `tokens_in/out`
- `estimated_cost`
- `latency_by_call`
- `provider_errors`
- `refusals`
- `schema_retries`
- `decision_reason`

La libreria debe integrarse con OpenTelemetry, logs estructurados y callbacks del usuario.

## Preguntas de Producto

### Identidad y Posicionamiento

1. El paquete se llamara `crupier`, `crupier-ai`, `ai-crupier` u otro nombre?
2. Queremos una marca en espanol ("Crupier", "baraja", "cartas") o una API internacional neutra?
3. Decision inicial: el producto principal sera SDK Python puro. Gateway/server local y CLI quedan como capas opcionales posteriores sobre el mismo core.
4. Debe competir con LiteLLM/Portkey/OpenRouter o integrarse con ellos?
5. El valor diferencial principal sera privacidad, calidad, coste, agentes o control local?

### Usuarios y Casos

6. Quien es el primer usuario serio: indie dev, empresa, equipo de agentes, consultora, investigador?
7. Que tipo de apps queremos soportar primero: chatbots, coding agents, RAG, backend batch, asistentes internos, data extraction?
8. Hay sectores regulados desde el inicio: legal, salud, finanzas, educacion, gobierno?
9. Se necesita multi-tenant desde el primer diseno?
10. Debe ser usable sin servidor, solo como libreria?

### API Publica

11. Queremos que la API imite OpenAI (`client.responses.create`) o tenga semantica propia (`crupier.deal`)?
12. La salida debe parecer una respuesta normal o siempre incluir trazas?
13. Streaming debe estar soportado tambien para fusion, aunque el juez llegue tarde?
14. Como se declaran constraints: YAML, Python objects, env vars, dashboard futuro?
15. Habra perfiles predefinidos: `cheap`, `fast`, `quality`, `private`, `research`, `agentic`, `regulated`?

### Orquestacion

16. Decision inicial: Crupier usara un orquestador desde el inicio, apoyado por reglas, constraints, benchmarks, evals y fichas de capacidades. No sera solo reglas estaticas ni solo un LLM sin restricciones.
17. Queremos permitir que un LLM vea el prompt para decidir ruta aunque el prompt sea sensible?
18. Como calculamos confidence sin autoengano?
19. Cuando debe fusionar y cuando no?
20. Que umbral convierte una tarea en "alto riesgo"?
21. Se permite lanzar modelos en paralelo para reducir latencia si aumenta coste?
22. Debe haber rutas deterministas para produccion?

### Proveedores

23. Los adapters nativos obligatorios son OpenAI, Anthropic, Google Gemini y Ollama?
24. OpenRouter debe ser adapter opcional para acceder a muchos modelos con una clave?
25. Se aceptaran proveedores OpenAI-compatible genericos?
26. Como se gestionan claves por proveedor y por tenant?
27. Queremos BYOK estricto, claves gestionadas por Crupier o ambos?
28. Como se bloquea un proveedor por region o constraints?

### Seguridad y Privacidad

29. Decision inicial: metadata operativa por defecto; guardar prompts/respuestas sera opcional y configurable por proyecto/ruta, pensado para depuracion, evals y mejora del router.
30. Necesitamos redaccion opt-in para trazas/evals o basta zero-log por defecto?
31. Debe el router impedir enviar datos sensibles a modelos sin ZDR?
32. Como se audita que un fallback no salto a un proveedor prohibido?
33. Que hacer si un modelo devuelve datos secretos presentes en contexto?
34. Debe haber allowlist de herramientas por modelo/ruta?
35. Necesitamos soporte HIPAA/GDPR desde el diseno conceptual?

### Coste y Negocio

36. Crupier debe calcular coste antes de ejecutar o solo despues?
37. Habra presupuestos por usuario/proyecto/request?
38. Debe abortar si el coste estimado supera limite?
39. Queremos optimizar por "calidad por euro" con datos reales del cliente?
40. Se puede vender como open source con plan enterprise futuro?

### Evaluacion

41. Que evals internas necesitamos antes de confiar en routing automatico?
42. Que dominios deben tener benchmarks propios?
43. Aceptamos jueces LLM o necesitamos validadores deterministas cuando haya schema/ground truth?
44. Como mediremos que fusion mejora sobre el mejor modelo individual?
45. Como detectaremos drift de modelos `latest`?

### Packaging y Operacion

46. Version minima de Python?
47. Dependencias duras vs extras: `crupier[openai]`, `crupier[anthropic]`, `crupier[google]`, `crupier[ollama]`?
48. Se debe publicar como libreria pura o tambien con server FastAPI opcional?
49. Licencia: MIT, Apache-2.0, BUSL, dual license?
50. Como se firmaran releases y se protegera supply chain?

## Edge Cases Criticos

- Un modelo del panel falla, rate limit, timeout o devuelve refusal.
- Dos modelos dan respuestas contradictorias pero ambas plausibles.
- El juez favorece al modelo mas verboso o convincente, no al correcto.
- La fusion cuesta mas y tarda mas que un frontier unico sin mejorar calidad.
- Un alias `latest` cambia y rompe comportamiento reproducible.
- Un proveedor depreca modelo con poco aviso.
- Un fallback salta a un proveedor no permitido por compliance.
- Un modelo no soporta `temperature`, `top_p`, tools, structured output o system messages.
- Tool schemas validos en un proveedor fallan en otro.
- Tokenizers distintos cambian costes y truncation.
- Long-context produce perdida silenciosa de informacion.
- Streaming de panel produce respuestas parciales dificiles de fusionar.
- Structured output invalido tras varias reparaciones.
- Instrucciones no confiables dentro de web/search/file content.
- Tools, URLs, callbacks o logs que cambian coste, latencia o comportamiento esperado.
- Herramientas web sin allowlist que rompen constraints del proyecto.
- Recursion infinita: fusion llama fusion o agente llama agente sin limite.
- Retry storm que multiplica coste y carga.
- Cost runaway por paneles grandes o tool loops.
- Race conditions en rutas paralelas.
- Cache devuelve respuesta vieja para contexto sensible.
- Un usuario pide "usa el modelo mas nuevo" pero los constraints exigen estable.
- Datos sensibles declarados enviados a proveedor con retencion no permitida.
- Fable/otros modelos con retencion obligatoria chocan con ZDR.
- Ollama Cloud requiere API key, o el host local configurado explicitamente no esta levantado.
- Ollama OpenAI-compatible no soporta una parte stateful de Responses API.
- Google preview/latest cambia con aviso y rompe evals.
- Region/proveedor no disponible temporalmente.
- Diferencias de refusal y comportamiento entre proveedores.
- Salida de reasoning/thinking no portable entre modelos.
- Archivos subidos tienen lifecycles distintos por proveedor.
- Los logs de observabilidad capturan prompts por accidente.
- Multi-tenant mezcla claves, trazas o budgets.
- Un modelo barato responde rapido pero con baja confianza; otro caro responde tarde con mejor solucion.
- La app cliente espera una respuesta OpenAI-like y no sabe procesar trazas ricas.

## Edge Cases de Exito

Estos casos no son solo errores tecnicos: son situaciones que pueden hacer que Crupier no genere confianza, aunque funcione.

### Confianza en el Orquestador

- El orquestador elige siempre el modelo "famoso" aunque otro modelo permitido rinda mejor en el dominio del proyecto.
- El orquestador sobreusa fusion porque "parece mas inteligente", disparando coste y latencia.
- El orquestador subestima tareas sencillas que contienen una instruccion critica oculta.
- El orquestador no sabe decir "no tengo suficientes datos para elegir bien".
- El orquestador cambia de criterio entre ejecuciones similares sin explicacion.
- El orquestador aprende de historico sesgado y refuerza malas rutas.
- El usuario no entiende por que se eligio una ruta y pierde confianza.
- El `DecisionTrace` es demasiado largo, tecnico o inutil para debugging real.

### Benchmarks y Evals

- Benchmarks publicos no representan el workload real del usuario.
- El sistema optimiza para benchmarks y empeora en produccion.
- Un modelo gana evals automaticas pero falla en criterios humanos de estilo, tono o utilidad.
- Un juez LLM puntua mejor respuestas largas aunque sean menos correctas.
- Evals locales contienen datos sensibles y no deben salir del entorno.
- Los datasets de eval quedan obsoletos cuando cambia el producto del usuario.
- Las evals no cubren refusals, tool errors, latencia, coste ni structured output.
- Comparar modelos sin controlar contexto, prompts, tools y temperatura produce resultados falsos.

### Actualizacion y Drift

- `crupier update` cambia recomendaciones y rompe expectativas sin revision humana.
- Una capability card queda stale y declara soportada una feature que el proveedor cambio.
- Cambia el precio de un modelo y rutas antes baratas se vuelven caras.
- Cambian rate limits o entitlements por cuenta y el modelo existe pero no es accesible.
- El proveedor cambia condiciones de datos y una ruta antes valida deja de serlo.
- Un modelo preview se comporta bien en evals y desaparece antes de produccion.
- El usuario usa alias `latest` en desarrollo y olvida fijar version antes de desplegar.

### UX y Adopcion

- Integrar Crupier requiere demasiados cambios respecto al SDK actual del usuario.
- El usuario no sabe si debe llamar `deal`, `responses.create`, `route` o `fusion`.
- La configuracion inicial es demasiado potente y abruma.
- El paquete instala demasiadas dependencias aunque el usuario solo quiera un proveedor.
- Los errores de claves/API keys no indican claramente que proveedor o modelo fallo.
- La primera experiencia con Ollama falla porque falta `OLLAMA_API_KEY`, el entitlement del modelo, o el host local configurado explicitamente no esta corriendo.
- El usuario espera que Crupier provea OpenRouter, cuando en realidad es BYOK opcional.
- Los perfiles `agentic`, `cheap`, `quality`, etc. no tienen semantica suficientemente clara.

### Compliance y Datos

- Un prompt con sensibilidad mal declarada se enruta a un proveedor no deseado.
- Redaccion de trazas/evals rompe calidad de debugging sin avisar.
- Guardar prompts/respuestas esta activado para evals y alguien lo usa en produccion sin TTL.
- Trazas compartidas en tickets internos contienen secretos.
- Fallback entre regiones incumple una restriccion contractual.
- Un proveedor subprocesa datos de forma distinta segun endpoint, region o plan.
- El usuario necesita borrar datos de un request concreto y Crupier no puede localizar trazas relacionadas.

### Operacion Real

- Timeouts parciales dejan al sistema sin saber si cobrar, reintentar o devolver respuesta incompleta.
- Una ruta paralela obtiene una respuesta buena tarde, despues de haber devuelto una peor.
- Un retry crea acciones duplicadas cuando hay tools con efectos secundarios.
- El sistema reintenta una refusal del proveedor sin clasificar si era capacidad, disponibilidad o constraint declarado.
- Un modelo devuelve JSON valido pero semanticamente incorrecto.
- Un provider outage provoca estampida hacia otro proveedor y agota su rate limit.
- La cache mejora coste pero oculta cambios recientes en datos o constraints.
- Multi-tenant mezcla historico de calidad y recomienda modelos usando datos de otro cliente.

### Agentes

- Un agente delega a Crupier, Crupier invoca tools, y el agente vuelve a delegar creando bucles.
- La eleccion de modelo por paso ignora el estado global del agente.
- El planificador usa un modelo fuerte, pero los subpasos baratos introducen errores silenciosos.
- El verificador usa el mismo sesgo/familia que el generador y no detecta fallos.
- Una herramienta requiere consentimiento humano, pero una ruta automatica intenta ejecutarla.
- El agente necesita reproducibilidad para auditoria y el router usa decisiones dinamicas no fijadas.
- Memoria o contexto compactado elimina la razon original de una decision de routing.

## Plan de Mitigacion de Edge Cases

Esta tabla convierte los edge cases de exito en requisitos de producto. Cada mitigacion debe poder probarse antes de considerar Crupier listo.

| Riesgo | Mitigacion | Criterio de exito |
| --- | --- | --- |
| El orquestador elige por fama, no por fit real | Separar candidatos permitidos, benchs publicos, evals locales e historico por perfil; obligar al orquestador a justificar contra esas senales. | `DecisionTrace` muestra senales usadas y por que el modelo elegido supera alternativas relevantes. |
| Sobreuso de fusion | Fusion requiere umbral: riesgo alto, incertidumbre alta, necesidad de consenso, investigacion o contradicciones. Coste/latencia estimados visibles antes de ejecutar si superan limites. | Tareas simples usan `single` o `cascade`; fusion solo aparece cuando el plan explica beneficio esperado. |
| El orquestador no sabe decir "no se" | Permitir salida `insufficient_signal` en `RoutePlan`; fallback a constraints deterministas o pedir confirmacion. | En escenarios sin datos suficientes no inventa confianza y devuelve advertencia accionable. |
| Decisiones inconsistentes | Versionar capability cards, prompt del orquestador, constraints y route profiles; permitir modo `deterministic`. | Dos requests equivalentes con mismo snapshot producen el mismo plan o explican la diferencia. |
| Historico sesgado | Separar historico por proyecto, tenant, perfil y dominio; usar decaimiento temporal; permitir reset/ignore history. | Una mala racha o un tenant no contamina recomendaciones globales. |
| Trace inutil o demasiado largo | Dos niveles de trace: `summary` para humanos y `debug` para auditoria completa. | Un usuario entiende en menos de 30 segundos por que se eligio la ruta. |
| Benchmarks publicos no representan el caso real | Benchmarks publicos son solo prior; evals locales tienen mayor peso cuando existen. | Una ruta puede cambiar tras evals locales aunque contradiga ranking publico. |
| Optimizacion excesiva para benchmarks | Medir coste, latencia, refusals, schema validity, tool success y ratings humanos, no solo score. | No se promueve una ruta si mejora score pero empeora restricciones clave. |
| Jueces LLM sesgados | Usar rubricas cortas, normalizar longitud, multi-judge opcional y validadores deterministas cuando haya ground truth. | Respuestas largas no ganan automaticamente; el score requiere evidencia/rubrica. |
| Evals con datos sensibles | Evals locales por defecto no salen del entorno; export requiere opt-in y redaccion. | `crupier eval run` puede ejecutarse offline/local sin enviar dataset a terceros salvo modelos configurados. |
| `crupier update` rompe rutas | Generar diff de cambios, changelog local y modo `--dry-run`; requerir confirmacion para aplicar cambios que afecten perfiles activos. | El usuario ve que rutas cambian antes de aceptar update. |
| Capability cards stale | Guardar `last_checked`, fuente y TTL; marcar capacidades inciertas; healthcheck opcional por modelo. | Si una ficha vence, Crupier advierte o revalida antes de usar features criticas. |
| Cambios de precio/rate limit | Estimaciones con fecha y fuente; budgets duros; circuit breakers por proveedor. | Una subida de coste no puede superar presupuesto configurado silenciosamente. |
| Preview/latest en produccion | `latest` y preview desactivados por defecto; profile prod exige IDs estables. | Un deploy en modo produccion falla si quedan aliases dinamicos no autorizados. |
| Integracion demasiado dificil | API principal minima (`Crupier.from_project().deal(...)`) y capa OpenAI-like opcional. Extras por proveedor. | Primer ejemplo funciona con menos de 20 lineas y una sola dependencia base. |
| Configuracion abruma | `crupier init` genera perfil simple; configuracion avanzada queda comentada o separada. | Un proyecto nuevo puede empezar con un proveedor y un perfil en minutos. |
| Errores de credenciales confusos | Errores tipados por proveedor/modelo/env var y sugerencia concreta. | El mensaje indica que clave falta, donde ponerla y que modelo intento usar. |
| Ollama no disponible | Preflight checks para host, modelo disponible y Cloud auth; fallback opcional. | Antes de ejecutar una ruta Ollama/local-first, Crupier detecta si el proveedor esta listo. |
| Confusion con OpenRouter | Docs y config lo etiquetan como `optional BYOK gateway`; nunca aparece en defaults. | Usuarios no interpretan que Crupier revende o provee OpenRouter. |
| Sensibilidad mal declarada | El usuario puede marcar sensibilidad/retencion por request o perfil; Crupier no actua como scanner de contenido por defecto. | La decision de routing respeta constraints declarados sin prometer deteccion universal. |
| Redaccion rompe calidad | Redaccion solo para trazas/evals opt-in; trace indica campos redaccionados; perfiles definen si abortar o continuar. | El usuario sabe que se redacciono y puede evaluar impacto. |
| Logging de contenido en produccion | `store_prompts/responses=false` por defecto; TTL obligatorio para `full`; warning en perfil prod. | No se puede activar almacenamiento completo en prod sin configuracion explicita. |
| Borrado/auditoria de datos | Trazas con IDs, indices por request y comandos de purge/export. | Un request concreto puede localizarse y purgarse si fue almacenado. |
| Fallback mal clasificado | Refusals clasificados; fallback solo si refusal es de capacidad/disponibilidad, no si viola constraints declarados. | Una negativa del proveedor no se reintenta automaticamente en otro modelo sin razon operativa clara. |
| Tools con efectos secundarios duplicados | Idempotency keys, tool call ledger y reglas de retry por tool. | Retries no ejecutan dos veces pagos, emails, writes o acciones externas. |
| JSON valido pero incorrecto | Validadores semanticos por dominio, constraints y tests post-output. | Una respuesta structured no se acepta solo por parsear. |
| Estampida por outage | Circuit breakers, backoff, limites por proveedor y fallback escalonado. | Un outage no redirige todo el trafico a un unico backup hasta agotarlo. |
| Cache oculta cambios recientes | Cache keys incluyen constraints/profile/model snapshot; TTL por sensibilidad; bypass para datos frescos. | Cambios de constraints o modelo invalidan cache relevante. |
| Multi-tenant leakage | Namespaces por tenant para claves, historico, traces, evals y budgets. | No hay recomendaciones ni trazas cruzadas entre tenants. |
| Bucles en agentes | `max_depth`, `max_calls`, `max_cost`, recursion headers y trace de delegacion. | Un agente no puede delegar indefinidamente a Crupier ni a subrutas. |
| Estado global ignorado en agentes | `AgentStateSummary` opcional en `RequestEnvelope`; decision por paso incluye objetivo global y riesgos acumulados. | El routing de subpasos considera plan completo, no solo prompt local. |
| Verificador comparte sesgo del generador | Preferir familias/proveedores distintos para critique/verify cuando los constraints lo permitan. | En `critique_repair`, el verificador no usa el mismo modelo por defecto. |
| Herramientas requieren humano | Tool policies con `requires_approval`; Crupier puede planear pero no ejecutar sin confirmacion. | Acciones sensibles quedan bloqueadas hasta aprobacion humana. |
| Reproducibilidad de auditoria | Snapshots de registry, route profile y policy por trace; modo `locked`. | Una decision pasada puede reproducirse o explicarse con su snapshot. |
| Compaction borra razon de routing | Guardar decision summary separada de la memoria conversacional. | Aunque se compacte el contexto del agente, la razon del plan queda en trace. |

## Requisitos de Privacidad y Operacion

### Defaults

- No guardar prompts ni respuestas por defecto.
- Metadata minima: modelo, proveedor, tokens, coste estimado, latencia, errores.
- Redaccion opt-in/opt-out configurable.
- Separar logs de decision de contenido sensible.
- No imprimir claves/API keys en errores.
- Validar URLs y herramientas antes de ejecutarlas.
- Timeouts y presupuestos obligatorios en paneles.
- Fallbacks constraint-aware, nunca "a cualquier cosa".

Esto no convierte Crupier en un gateway de seguridad. Son requisitos de higiene operativa para que el SDK pueda moverse entre modelos sin filtrar claves, gastar sin limite o perder reproducibilidad.

### Riesgos Operativos Iniciales

| Riesgo | Mitigacion requerida |
| --- | --- |
| Instrucciones no confiables en web/search/file content | Separacion instrucciones/datos, resumen de contexto y constraints de herramientas. |
| Secretos en logs o errores | Redaction para trazas opt-in, no logs por defecto, no imprimir claves/API keys. |
| Cost runaway | Max calls/tokens/coste, circuit breakers, retry budget. |
| Proveedor no permitido | Constraint engine antes de RoutePlan y antes de cada fallback. |
| Supply chain PyPI | 2FA, trusted publishing, firmas/provenance, lockfiles de dev, dependencias minimas. |
| Cross-tenant leakage | Context isolation, credential scoping, trace partitioning. |
| Drift de modelos | Registry snapshots, evals recurrentes, alerts de alias/deprecacion. |
| Respuesta incorrecta de juez | Multi-judge opcional, quorum, validadores deterministicos, benchmarks. |

## Decisiones Propuestas para v1 de Producto

Estas no son implementacion; son apuestas iniciales para validar contigo.

1. Crupier debe ser primero una libreria Python pura, con server/gateway opcional despues. El core debe quedar desacoplado para poder usarse en agentes, apps web, jobs batch, notebooks, CLIs y futuros servicios.
2. La API publica debe ser propia pero amigable para usuarios OpenAI-like.
3. Modelos estables fijados por defecto; aliases `latest` solo con flag explicito.
4. OpenAI, Anthropic, Google y Ollama son adapters de primera clase.
5. OpenRouter debe ser adapter opcional BYOK, no dependencia central ni ruta recomendada por defecto. Crupier no proveera OpenRouter; si se usa, lo configura el usuario con su cuenta, sus claves y sus condiciones.
6. La fusion debe devolver respuesta final y analisis estructurado util para comparar calidad, coste, latencia y riesgo operativo.
7. El router debe ser constraints-first: filtrar por capacidades requeridas, modelos permitidos, presupuesto, latencia, modalidad y parametros soportados antes de optimizar calidad. Crupier no es un gateway de seguridad ni un inspector de contenido.
8. Evals y trazas no son extras; forman parte del producto.
9. No se expondra chain-of-thought privada; solo resumen/analisis permitido.
10. La configuracion debe poder vivir en Python y en YAML/TOML.
11. Crupier debe tener orquestador desde el inicio: un modelo orquestador que decide usando constraints, benchs, registry, historico y evals, no solo una tabla fija de reglas.
12. El usuario debe poder definir los modelos candidatos del proyecto y ejecutar un update que regenere capacidades, perfiles y recomendaciones de routing.
13. Crupier debe tener adopcion drop-in para proyectos existentes: proxy OpenAI-compatible, autopatch opt-in, clientes compatibles y SDK nativo sobre el mismo core.

### Decision: SDK primero, server opcional

Recomendacion: empezar por SDK Python puro.

Razon: si el producto nace para agentes pero no quiere cerrarse a ellos, el nucleo debe poder vivir dentro del proceso del usuario. Un agente, una app FastAPI, un worker Celery, un notebook, un CLI o una herramienta interna deberian poder importar `crupier` y ejecutar el mismo motor de routing sin depender de un servidor aparte.

El server/gateway sigue siendo importante, pero como capa posterior. Debe envolver el mismo core para casos multi-lenguaje, equipos enterprise, control centralizado de claves, presupuestos por tenant, observabilidad compartida y despliegues internos. No debe ser el centro conceptual del producto.

Implicacion arquitectonica: desde el primer diseno, `RoutePlanner`, `PolicyProfile`, `ModelRegistry`, `ProviderAdapter`, `RouteExecutor` y `DecisionTrace` deben ser objetos serializables y transportables. Asi el SDK no queda atrapado en Python cuando llegue el momento de exponer HTTP, CLI o plugins.

### Decision: adopcion drop-in

Recomendacion: Crupier debe poder integrarse en proyectos existentes por capas.

Niveles:

- `proxy`: cambiar `OPENAI_BASE_URL` hacia un server OpenAI-compatible de Crupier.
- `autopatch`: una linea `crupier.install()` o variable `CRUPIER_AUTOPATCH`.
- `compatible client`: cambiar imports a clientes compatibles mantenidos por Crupier.
- `native SDK`: usar `Crupier.from_project().deal(...)`.

El objetivo comercial es claro: un usuario que descarga un repo de GitHub no deberia tener que reescribir la aplicacion para probar Crupier. Si el proyecto usa un SDK o endpoint comun, Crupier debe poder ponerse delante, interpretar la llamada y mejorar routing, coste, latencia, fallback y trazas sin romper la forma de respuesta.

Limite honesto: "sin cambiar nada" solo es viable cuando el proyecto permite configurar endpoint, entorno o import. Si el codigo fija clientes, claves, modelos o response classes de forma rigida, Crupier debe ofrecer rutas de integracion minima y errores claros.

### Decision: orquestador desde el inicio

Recomendacion: Crupier debe tener un modelo orquestador desde el inicio, pero con validacion estructurada.

El orquestador no debe decidir solo por prompt. Debe recibir un contexto estructurado: modelos permitidos, capacidades conocidas, constraints del proyecto, coste estimado, latencia historica, resultados de benchmarks publicos, resultados de evals propias y requisitos de la peticion. Con eso produce un `RoutePlan` explicable.

Arquitectura recomendada:

- Reglas duras: proveedor/modelo permitido, presupuesto maximo, latencia, modalidad requerida, capacidades verificadas y parametros soportados.
- Registry: fichas de modelos y capacidades actualizadas.
- Benchmarks: senales externas y locales sobre fortalezas/debilidades.
- Orquestador LLM: decide entre candidatos validos y explica la decision.
- Validador: revisa que el plan no viole constraints del proyecto antes de ejecutar.
- Feedback loop: resultados reales alimentan historico, evals y recomendaciones futuras.

Esto evita dos extremos malos: un router rigido que se queda viejo y un LLM que improvisa. Crupier debe ser mas parecido a un director tecnico con datos que a una ruleta.

### Decision: update de modelos y fichas del proyecto

Recomendacion: el usuario define que modelos quiere permitir y Crupier actualiza las fichas del proyecto.

Flujo conceptual:

1. El usuario declara modelos/proveedores candidatos en `crupier.toml` o Python.
2. Ejecuta `crupier update`.
3. Crupier consulta fuentes oficiales/APIs disponibles, precios, deprecaciones y capacidades.
4. Crupier actualiza un registry local del proyecto.
5. Crupier puede correr evals locales opcionales contra esos modelos.
6. Crupier genera recomendaciones de routing y perfiles tipo `agentic`, `cheap`, `quality`, `private`, `research`.

Nombre sugerido para evitar confusion: llamar a esas fichas `model cards`, `capability cards` o `route profiles`, no necesariamente "skills". Si luego queremos integrarlo con sistemas de agentes que ya usan skills, se puede exportar a ese formato como capa adicional.

## Modelo Conceptual

- `ProviderAdapter`: cliente normalizado por proveedor.
- `ModelRegistry`: catalogo versionado de capacidades.
- `CapabilityCard`: ficha local por modelo/proveedor con capacidades, costes, restricciones, benchs y notas de compatibilidad.
- `PolicyProfile`: privacidad, coste, region, modalidad, proveedores permitidos.
- `RequestEnvelope`: input normalizado.
- `RoutePlanner`: crea plan antes de ejecutar.
- `RouteExecutor`: ejecuta single/fallback/panel/fusion.
- `Judge`: compara y sintetiza sin exponer razonamiento privado.
- `Validator`: comprueba schema, factualidad, constraints y coherencia de la ruta.
- `DecisionTrace`: auditoria de por que se eligio una ruta.
- `EvalHarness`: prueba rutas contra datasets.

## Producto No-MVP: Definicion de "Listo"

Crupier no deberia considerarse producto listo hasta tener:

- Adapters robustos para los cuatro proveedores objetivo.
- Registry de modelos versionado y actualizable.
- Constraints de proyecto aplicados antes de cada llamada.
- Fallbacks y timeouts probados.
- Fusion con proteccion de recursion y presupuesto.
- Structured outputs y tools con compatibilidad documentada.
- Observabilidad completa.
- Evals reproducibles.
- Documentacion de edge cases.
- Guia de privacidad, data retention y coste.
- Pipeline de release seguro para PyPI.

## Proximos Pasos de Discovery

1. Responder las preguntas de producto prioritarias.
2. Elegir nombre de paquete y posicionamiento.
3. Definir 5 casos de uso canonicos.
4. Crear matriz de capacidades proveedor/modelo.
5. Diseñar defaults de privacidad y retencion.
6. Definir API publica en pseudocodigo, sin implementarla.
7. Definir estructura del documento tecnico de arquitectura.
8. Preparar modelo de riesgos operativos.
9. Preparar plan de evals.
10. Solo despues, decidir fase de implementacion.
