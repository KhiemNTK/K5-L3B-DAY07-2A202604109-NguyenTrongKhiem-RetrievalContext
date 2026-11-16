"""Aggregate source-reviewed labels; never infer grounding from citation IDs."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from baseline import TfidfIndex, hit


def aggregate(cases, traces, reviews):
    """Cases: id/answerable; traces: query_id/status/events/citations.

    Reviews: query_id, answer_grade (0..2 for answerable), citation_support
    (one boolean per citation), and abstention_correct (boolean for unanswerable).
    Provider errors are excluded and reported.
    """
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('case ID trùng')
    if len({r['query_id'] for r in traces}) != len(traces) or len({r['query_id'] for r in reviews}) != len(reviews):
        raise ValueError('trace/review ID trùng')
    trace_by_id = {r['query_id']: r for r in traces}
    review_by_id = {r['query_id']: r for r in reviews}
    out = {'cases': len(cases), 'provider_errors': 0, 'tool_use': {'n': 0, 'N': 0},
           'grounded_answer': {'0': 0, '1': 0, '2': 0, 'N': 0, 'mean': None},
           'citation_validity': {'n': 0, 'N': 0}, 'answer_citation_coverage': {'n': 0, 'N': 0},
           'abstention': {'n': 0, 'N': 0}, 'false_abstention': {'n': 0, 'N': 0}}
    for case in cases:
        cid = case['id']
        if not isinstance(case.get('answerable'), bool): raise ValueError(f'{cid}: thiếu answerable boolean')
        if cid not in trace_by_id: raise ValueError(f'{cid}: thiếu trace')
        t = trace_by_id[cid]
        if t.get('status') == 'error':
            out['provider_errors'] += 1
            continue
        if t.get('status') not in ('answered', 'abstain'): raise ValueError(f'{cid}: status không hợp lệ')
        if cid not in review_by_id: raise ValueError(f'{cid}: thiếu nhãn audit')
        r = review_by_id[cid]
        events = t.get('events')
        citations = t.get('citations')
        support = r.get('citation_support')
        if not isinstance(events, list) or not isinstance(citations, list): raise ValueError(f'{cid}: events/citations không hợp lệ')
        if not isinstance(support, list) or len(support) != len(citations) or any(type(v) is not bool for v in support):
            raise ValueError(f'{cid}: cần một nhãn boolean cho mỗi citation')
        out['tool_use']['N'] += 1
        out['tool_use']['n'] += any(e.get('event') == 'search' for e in events)
        out['citation_validity']['n'] += sum(support)
        out['citation_validity']['N'] += len(citations)
        if t['status'] == 'answered':
            out['answer_citation_coverage']['N'] += 1
            out['answer_citation_coverage']['n'] += bool(citations)
        if case['answerable']:
            grade = r.get('answer_grade')
            if type(grade) is not int or grade not in (0, 1, 2): raise ValueError(f'{cid}: answer_grade phải là 0, 1 hoặc 2')
            if t['status'] == 'abstain' and grade != 0: raise ValueError(f'{cid}: abstain trên câu có đáp án phải được chấm 0')
            out['grounded_answer'][str(grade)] += 1
            out['grounded_answer']['N'] += 1
            out['false_abstention']['N'] += 1
            out['false_abstention']['n'] += t['status'] == 'abstain'
        else:
            correct = r.get('abstention_correct')
            if type(correct) is not bool:
                raise ValueError(f'{cid}: cần nhãn abstention_correct do người chấm xác nhận')
            if correct and t['status'] != 'abstain':
                raise ValueError(f'{cid}: nhãn từ chối đúng không khớp status')
            out['abstention']['N'] += 1
            out['abstention']['n'] += correct
    g = out['grounded_answer']
    if g['N']: g['mean'] = (g['1'] + 2*g['2']) / g['N']
    return out


def evaluate_run(chunks, cases, traces, reviews):
    """Combine source-reviewed labels with original-query TF-IDF ranking."""
    ids = {c['id'] for c in cases}
    if {t['query_id'] for t in traces} != ids or {r['query_id'] for r in reviews} != ids:
        raise ValueError('Review the complete run, including errors; do not select a subset')
    report = aggregate(cases, traces, reviews)
    by_trace = {t['query_id']: t for t in traces}
    by_review = {r['query_id']: r for r in reviews}
    index = TfidfIndex(chunks)
    originals, strict, context_counts, grounded_sum = [], 0, {'0': 0, '1': 0, '2': 0}, 0
    answerable_count = sum(c['answerable'] for c in cases)
    for case in cases:
        trace, review = by_trace[case['id']], by_review[case['id']]
        if trace.get('method') != 'tfidf':
            raise ValueError('Original-query ranking in this evaluator requires TF-IDF runs')
        if not isinstance(review.get('reason'), str) or not review['reason'].strip():
            raise ValueError(f"{case['id']}: a source-review reason is required")
        if case['answerable']:
            rows = index.search(case['question'], top_k=3, filters=case.get('filter'))
            originals.append({'query_id': case['id'], 'query': case['question'], 'filter': case.get('filter'),
                              'results': rows, 'hit1': bool(rows and hit(rows[0], case['gold'])),
                              'hit3': any(hit(row, case['gold']) for row in rows)})
            grade = review.get('context_grade')
            if grade is None and trace['status'] == 'error' and not trace.get('context_ids'):
                grade = 0
            if type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError('Every retrieved answerable context needs a grade 0..2, including model errors')
            if grade and not trace.get('context_ids'):
                raise ValueError('A positive context grade requires evidence in the actual context')
            context_counts[str(grade)] += 1
        if trace['status'] == 'error':
            continue
        if any(supported and cid not in trace['context_ids'] for cid, supported in
               zip(trace['citations'], review['citation_support'])):
            raise ValueError('A supported citation must exist in the actual final context')
        if case['answerable']:
            grounded_sum += review['answer_grade']
            strict += (trace['status'] == 'answered' and review['answer_grade'] == 2 and grade == 2
                       and bool(trace['citations']) and all(review['citation_support']))
        else:
            strict += review['abstention_correct']
    def fraction(n, total):
        return {'n': n, 'N': total, 'rate': n / total if total else None}
    report.update(strict_pass=fraction(strict, len(cases)), grounded_sum=grounded_sum,
                  grounded_mean_including_errors=grounded_sum / answerable_count if answerable_count else None,
                  hit1=fraction(sum(t['hit1'] for t in originals), answerable_count),
                  hit3=fraction(sum(t['hit3'] for t in originals), answerable_count),
                  context_evidence_rate=fraction(context_counts['2'], answerable_count),
                  context_grades=context_counts, original_query_traces=originals,
                  search_call_distribution=dict(Counter(
                      sum(e['event'] == 'search' for e in t.get('events', [])) for t in traces)),
                  median_ms=statistics.median(t['elapsed_ms'] for t in traces))
    return report


def main():
    parser = argparse.ArgumentParser(description='Tổng hợp trace agent và nhãn audit thủ công')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunks', type=Path, help='Add reviewed context and original-query TF-IDF metrics')
    args = parser.parse_args()
    def load(path): return json.loads(path.read_text(encoding='utf-8'))
    cases = [c for c in load(args.cases) if c.get('split') == args.split]
    if not cases: raise ValueError('không có case trong split')
    traces, reviews = load(args.traces), load(args.reviews)
    metrics = evaluate_run(load(args.chunks), cases, traces, reviews) if args.chunks else aggregate(cases, traces, reviews)
    report = {'split': args.split, **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'original_query_traces'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
