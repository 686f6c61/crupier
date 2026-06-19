# Crupier Real Smoke Test

Fecha: 2026-06-18  
Tipo: prueba real con API keys en variables de entorno temporales.  
Persistencia: no se guardaron claves, prompts ni respuestas completas.

## Resultado

| Proveedor | Discover | Modelos descubiertos | Modelo probado | Smoke | Latencia |
| --- | --- | ---: | --- | --- | ---: |
| OpenAI | OK | 126 | `openai:gpt-5.4-mini` | OK | 2414 ms |
| Anthropic Claude | OK | 8 | `anthropic:claude-opus-4-8` | OK | 1203 ms |
| Ollama Cloud | OK | 35 | `ollama:deepseek-v3.1:671b` | OK | 2040 ms |

## Update Online

Se valido `crupier update --online` contra proveedores reales en un proyecto temporal.

| Operacion | Resultado |
| --- | --- |
| Modelos descubiertos/escritos como capability cards | 169 |
| Archivos escritos en proyecto temporal | 170 |
| Warnings | 0 |

El proyecto temporal se elimino despues de la prueba.

## Prompt de Smoke

```text
Smoke test. Reply with exactly: crupier-ok
```

## Observaciones

- Las tres credenciales funcionaron para descubrir modelos y ejecutar una llamada real minima.
- OpenAI respondio correctamente via Responses API.
- Anthropic respondio correctamente via Messages API.
- Ollama Cloud respondio correctamente via API nativa `https://ollama.com/api`.
- No se imprimieron ni guardaron API keys.
- El venv temporal usado para instalar SDKs fue eliminado despues de la prueba.
