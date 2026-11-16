"""Run live dev experiments or a frozen full-test comparison."""
import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from agentic import CONTEXT_POLICY, run_agent
from baseline import TfidfIndex
from benchmark import load_cases, verify_lock
from controller import OpenRouterController
from day6 import ROOT, LabError, connect_day6, digest, error_details, load_snapshot
from experiments import code_hashes, load_selection, resolve_config
from retrieval import HybridRetriever


def usage_totals(calls):
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    costs = [call.get('usage', {}).get('cost') for call in calls]
    tokens = [call.get('usage', {}).get('total_tokens') for call in calls]
    return {'model_calls': len(calls),
            'total_tokens': sum(tokens) if calls and all(number(n) for n in tokens) else None,
            'service_cost_usd': sum(costs) if calls and all(number(n) for n in costs) else None,
            'reported_cost_usd': sum(n for n in costs if number(n))}


def validate_config(config):
    if not isinstance(config.get('model'), str) or not config['model'].strip():
        raise LabError('Agent model must be configured')
    if config.get('retrieval') not in ('dense', 'tfidf', 'hybrid'):
        raise LabError('Agent retrieval must be dense, tfidf or hybrid')
    for name in ['max_searches', 'top_k', 'context_budget_chars', 'max_output_tokens', 'timeout_seconds']:
        if type(config.get(name)) is not int or config[name] < 1:
            raise LabError(f'Agent configuration requires a positive integer: {name}')


def run_config(args):
    if args.split == 'test':
        if not args.selection or not args.variant or args.case or args.method or args.experiment:
            raise LabError('Test requires --selection and --variant, with no case/config overrides')
        return load_selection(args.selection, args.variant)
    if args.selection or args.variant:
        raise LabError('Selection and variant are only used for test')
    config = resolve_config(args.experiment or 'baseline')
    if args.method:
        config['retrieval'] = args.method
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6-root', required=True)
    parser.add_argument('--case', action='append', help='Dev ID; repeat to select smoke cases, omit to run all dev')
    parser.add_argument('--method', choices=['dense', 'tfidf', 'hybrid'])
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--experiment', help='Declared dev experiment; default baseline')
    parser.add_argument('--selection', type=Path, help='Frozen selection.json required for test')
    parser.add_argument('--variant', choices=['baseline', 'selected'])
    args = parser.parse_args()
    output = ROOT / 'results/agent' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + args.split)
    output.mkdir(parents=True)
    report, traces = {'split': args.split, 'experiment': args.experiment or 'baseline',
                      'variant': args.variant}, []
    phase = 'setup'
    try:
        from dotenv import dotenv_values
        config = run_config(args)
        validate_config(config)
        key = os.environ.get('OPENROUTER_API_KEY') or dotenv_values(ROOT / '.env').get('OPENROUTER_API_KEY')
        if not key:
            raise LabError('OPENROUTER_API_KEY is missing; configure the Day 7 .env file')
        cases = load_cases(args.split)
        if args.case:
            by_id = {case['id']: case for case in cases}
            if len(set(args.case)) != len(args.case) or any(cid not in by_id for cid in args.case):
                raise LabError('Requested cases must be unique dev IDs')
            cases = [by_id[cid] for cid in args.case]
        report.update(config=config, context_policy=CONTEXT_POLICY, case_ids=[case['id'] for case in cases],
                      input_sha256=verify_lock(args.split), agent_config_sha256=digest(ROOT / 'config/agent.json'),
                      experiments_sha256=digest(ROOT / 'config/experiments.json'), code_sha256=code_hashes())
        if args.selection:
            report.update(selection=args.selection.resolve().relative_to(ROOT).as_posix(),
                          selection_sha256=digest(args.selection))
        started = perf_counter()
        _, chunks = load_snapshot(Path(args.day6_root).resolve())
        lexical = TfidfIndex(chunks)
        if config['retrieval'] == 'tfidf':
            retriever = lexical
        else:
            dense, report['verification'] = connect_day6(args.day6_root)
            plan = json.loads((ROOT / 'benchmark/plan.json').read_text(encoding='utf-8'))['retrieval']
            retriever = dense if config['retrieval'] == 'dense' else HybridRetriever(
                dense, lexical, candidate_k=plan['hybrid_candidate_k'], rrf_k=plan['rrf_k'])
        report['setup_seconds'] = perf_counter() - started
        phase = 'agent'
        for case in cases:
            controller = OpenRouterController(key, model=config['model'], max_searches=config['max_searches'],
                max_tokens=config['max_output_tokens'], timeout=config['timeout_seconds'])
            trace = {'query_id': case['id'], 'split': args.split, 'question': case['question'],
                     'filter': case.get('filter'), 'method': config['retrieval']}
            started = perf_counter()
            try:
                trace.update(run_agent(case['question'], controller, retriever, filters=case.get('filter'),
                    max_searches=config['max_searches'], top_k=config['top_k'],
                    budget=config['context_budget_chars'], trace=trace))
            except Exception as error:
                trace.update(status='error', phase='agent', **error_details(error))
            trace.update(model_calls=controller.calls, usage=usage_totals(controller.calls),
                         elapsed_ms=(perf_counter() - started) * 1000)
            traces.append(trace)
            (output / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8', newline='\n')
            print(json.dumps({'query_id': case['id'], 'status': trace['status'],
                              'model_calls': len(controller.calls)}), flush=True)
        calls = [call for trace in traces for call in trace['model_calls']]
        report.update(status='error' if any(t['status'] == 'error' for t in traces) else 'completed',
                      outcomes=dict(Counter(t['status'] for t in traces)), usage=usage_totals(calls))
    except Exception as error:
        report.update(status='error', phase=phase, **error_details(error))
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                         encoding='utf-8', newline='\n')
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'outcomes': report.get('outcomes'), 'usage': report.get('usage')}, indent=2))
    if report['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
