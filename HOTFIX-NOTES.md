# Hotfix notes

## gpt-4o-mini → gpt-6-luna

OpenAI is shutting down `gpt-4o` and `gpt-4o-mini` on Jan 14, 2027. Every chat model in the
course code now uses `gpt-6-luna`. Embeddings stay on `text-embedding-3-small` (not deprecated).

`gpt-6-luna` is a reasoning model. Every call passes `reasoning_effort="none"` so it stays fast
and cheap and accepts `temperature`. Leave it out and you pay for hidden reasoning tokens.

### What the videos show vs what the code has now

| File | In the video | In the code now |
|---|---|---|
| `lang_test.py` | `init_chat_model("gpt-4o-mini")` (and the commented `ChatOpenAI` / `init_chat_model(...)` variants) | `init_chat_model("gpt-6-luna", reasoning_effort="none")`; commented variants updated the same way |
| `vector_db_llm.py` | `client.chat.completions.create(model="gpt-4o-mini", ...)` | `model="gpt-6-luna", reasoning_effort="none"` |
| `vector_db_llm_lang_chain.py` | `init_chat_model("gpt-4o-mini")` | `init_chat_model("gpt-6-luna", reasoning_effort="none")` |
| `pinecone_db_llm_lang_chain.py` | `ChatOpenAI(api_key=..., model="gpt-4o-mini")` | `ChatOpenAI(api_key=..., model="gpt-6-luna", reasoning_effort="none")` |
| `raw_emb.py` | commented chat example with `gpt-4o-mini` | commented example now `gpt-6-luna` + `reasoning_effort="none"` (embedding code unchanged) |
| `pg_vector_section/5-langchain/04_rag_chain.py` | `ChatOpenAI(model="gpt-4o-mini", temperature=0)` | `ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)` |
| `pg_vector_section/6-supabase/03_production_example.py` | `Config.chat_model = "gpt-4o-mini"` | `Config.chat_model = "gpt-6-luna"`, new `Config.reasoning_effort = "none"` passed to `ChatOpenAI` |
| `README.md` | "now using `gpt-4o-mini`" | "now using `gpt-6-luna`" |

Answers read a little differently from the ones on screen. Same retrieval, a newer model writing the answer.

If you call the raw OpenAI SDK yourself: `gpt-6-luna` rejects `stop`, and wants
`max_completion_tokens` instead of `max_tokens`. LangChain's `max_tokens` is mapped for you.

### Tested
- Live against the OpenAI API: `lang_test.py`, `vector_db_llm.py`, `vector_db_llm_lang_chain.py`,
  `raw_emb.py`, `pinecone_db_llm_lang_chain.py` (against the existing Pinecone index).
- `04_rag_chain.py`: chain run end to end with a Chroma store standing in for Postgres.
- `03_production_example.py`: import-checked, and its exact LLM config run live. Not run
  end to end against Supabase.
