"""Compare dense, TF-IDF and rank fusion on one benchmark split."""
import argparse
import json
import statistics
from datetime import datetime, timezone
from time import perf_counter

from baseline import TfidfIndex, hit, pack_context, validate_search
from benchmark import load_cases, verify_lock
from day6 import ROOT, connect_day6, digest, error_details


class HybridRetriever:
    def __init__(self, dense, lexical, *, candidate_k=10, rrf_k=60):
        if type(candidate_k) is not int or candidate_k < 1 or type(rrf_k) is not int or rrf_k < 1:
            raise ValueError('candidate_k and rrf_k must be positive integers')
        self.dense, self.lexical = dense, lexical
        self.candidate_k, self.rrf_k = candidate_k, rrf_k
        self.last_steps = []

    def search(self, query, *, filters=None, top_k=3):
        self.last_steps = []
        validate_search(query, top_k, filters)
        fused = {}
        for name, retriever in [('dense', self.dense), ('tfidf', self.lexical)]:
            cutoff = max(self.candidate_k, top_k)
            step = {'method': name, 'query': query, 'filter': filters, 'top_k': cutoff}
            self.last_steps.append(step)
            started = perf_counter()
            try:
                rows = retriever.search(query, filters=filters, top_k=cutoff)
                if len(rows) > cutoff or len({row['id'] for row in rows}) != len(rows):
                    raise ValueError('A component returned too many results or duplicate chunk IDs')
            except Exception as error:
                step.update(status='error', **error_details(error), elapsed_ms=(perf_counter() - started) * 1000)
                raise
            step.update(status='ok', results=rows, elapsed_ms=(perf_counter() - started) * 1000)
            for rank, row in enumerate(rows, 1):
                cid = row['id']
                if cid not in fused:
                    fused[cid] = {**row, 'score': 0, 'strategy': 'hybrid', 'components': {}}
                fused[cid]['score'] += 1 / (self.rrf_k + rank)
                fused[cid]['components'][name] = {'rank': rank, 'score': row['score']}
        ranked = sorted(fused.values(), key=lambda row: (-row['score'], row['id']))[:top_k]
        return [{**row, 'rank': rank} for rank, row in enumerate(ranked, 1)]


def evaluate(retriever, cases, *, top_k, budget):
    """Gold is used only after search returns, to score the saved results."""
    if type(top_k) is not int or top_k < 3 or type(budget) is not int or budget < 1:
        raise ValueError('Comparison requires top_k >= 3 and a positive character budget')
    traces = []
    for case in cases:
        started = perf_counter()
        try:
            rows = retriever.search(case['question'], filters=case.get('filter'), top_k=top_k)
            context, ids = pack_context(rows, budget=budget)
            trace = {'status': 'ok', 'results': rows, 'context': context, 'context_ids': ids,
                     'context_chars': len(context), 'steps': getattr(retriever, 'last_steps', [])}
            if case['answerable']:
                trace.update(hit1=bool(rows and hit(rows[0], case['gold'])),
                             hit3=any(hit(row, case['gold']) for row in rows[:3]),
                             context_hit=any(row['id'] in ids and hit(row, case['gold']) for row in rows))
        except Exception as error:
            trace = {'status': 'error', **error_details(error),
                     'steps': getattr(retriever, 'last_steps', [])}
        trace.update(query_id=case['id'], split=case['split'], query=case['question'],
                     filter=case.get('filter'), top_k=top_k, elapsed_ms=(perf_counter() - started) * 1000)
        traces.append(trace)
    return traces


def summarize(cases, traces):
    answerable = {case['id'] for case in cases if case['answerable']}
    successful = [trace for trace in traces if trace['status'] == 'ok']
    report = {'cases': len(cases), 'successful': len(successful), 'errors': len(traces) - len(successful),
              'median_ms': statistics.median(t['elapsed_ms'] for t in traces), 'service_cost_usd': 0}
    for name, key in [('hit1', 'hit1'), ('hit3', 'hit3'), ('packed_context_evidence_preview', 'context_hit')]:
        n = sum(bool(t.get(key)) for t in traces if t['query_id'] in answerable)
        report[name] = {'n': n, 'N': len(answerable), 'rate': n / len(answerable) if answerable else None}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6-root', required=True)
    parser.add_argument('--split', choices=['dev'], required=True)
    args = parser.parse_args()
    cases = load_cases(args.split)
    plan = json.loads((ROOT / 'benchmark/plan.json').read_text(encoding='utf-8'))
    output = ROOT / 'results/retrieval' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + args.split)
    output.mkdir(parents=True)
    report = {'split': args.split, 'purpose': 'retrieval_comparison_not_agent_evaluation',
              'input_sha256': verify_lock(args.split), 'config': plan['retrieval'],
              'code_sha256': {name: digest(ROOT / 'starter' / name) for name in
                              ['retrieval.py', 'benchmark.py', 'baseline.py', 'day6.py']}}
    phase = 'connect_and_verify'
    try:
        started = perf_counter()
        dense, verification = connect_day6(args.day6_root)
        chunks = json.loads((ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
        lexical = TfidfIndex(chunks)
        config = plan['retrieval']
        hybrid = HybridRetriever(dense, lexical, candidate_k=config['hybrid_candidate_k'], rrf_k=config['rrf_k'])
        report.update(setup_seconds=perf_counter() - started, verification=verification, methods={})
        for name, retriever in [('dense', dense), ('tfidf', lexical), ('hybrid', hybrid)]:
            phase = name
            traces = evaluate(retriever, cases, top_k=config['top_k'], budget=config['context_budget_chars'])
            (output / f'{name}.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8', newline='\n')
            report['methods'][name] = summarize(cases, traces)
        report['status'] = 'error' if any(row['errors'] for row in report['methods'].values()) else 'completed'
    except Exception as error:
        report.update(status='error', phase=phase, **error_details(error))
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                         encoding='utf-8', newline='\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'methods': report.get('methods')}, indent=2))
    if report['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
