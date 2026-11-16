"""Bounded retrieval loop with whole-chunk context and citation checks."""
import re
from time import perf_counter
from typing import Any, Protocol
from baseline import pack_context

CONTEXT_POLICY = 'latest_search_first_then_previous; deduplicate_by_id; whole_chunks'

class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']},
        or {'action': 'abstain'} when retrieved evidence is insufficient.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800,
              trace: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if (not isinstance(question, str) or not question.strip()
            or any(type(n) is not int or n < 1 for n in (max_searches, top_k, budget))):
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    trace = {} if trace is None else trace
    trace.update(events=[], context='', context_ids=[], context_policy=CONTEXT_POLICY, budget_chars=budget)
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events = trace['events']

    def finish(status, answer, citations, reason):
        events.append({'event': 'answer' if status == 'answered' else 'abstain', 'reason': reason})
        return {**trace, 'status': status, 'answer': answer, 'citations': citations, 'stop_reason': reason,
                'citation_sources': [{k: chunks[cid][k] for k in ('id', 'source_id', 'source', 'page')}
                                     for cid in citations]}

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return finish('abstain', 'Không đủ bằng chứng để trả lời.', [], 'search_limit')
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            event = {'event': 'search', 'query': query, 'filter': filters, 'top_k': top_k,
                     'tool_call_id': action.get('tool_call_id')}
            events.append(event)
            started = perf_counter()
            try:
                results = retriever.search(query, filters=filters, top_k=top_k)
                if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                    raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            except Exception as error:
                event.update(status='error', error_type=type(error).__name__)
                raise
            finally:
                event['elapsed_ms'] = (perf_counter() - started) * 1000
            fresh_ids = {c['id'] for c in results}
            candidates = results + [c for c in candidates if c['id'] not in fresh_ids]
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            trace.update(context=context, context_ids=ids)
            event.update(status='ok', results=results, steps=getattr(retriever, 'last_steps', []),
                         context=context, context_ids=ids,
                         dropped_ids=[c['id'] for c in candidates if c['id'] not in ids])
            # Send only the latest cumulative context; avoid multiplying the budget.
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            markers = re.findall(r'\[([^\[\]]+)\]', answer)
            valid = bool(answer.strip() and citations and len(citations) == len(set(citations))
                         and set(markers) == set(citations) and all(i in chunks for i in citations))
            if valid:
                return finish('answered', answer, citations, 'model_answer')
            return finish('abstain', 'Không đủ bằng chứng để trả lời.', [], 'invalid_or_missing_citations')
        elif action.get('action') == 'abstain':
            return finish('abstain', 'Không đủ bằng chứng để trả lời.', [], 'model_abstain')
        else:
            raise ValueError('action phải là search, answer hoặc abstain')
    return finish('abstain', 'Không đủ bằng chứng để trả lời.', [], 'model_call_limit')
