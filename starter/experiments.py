"""Select on fully reviewed dev runs and freeze the inputs used by test."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agent_metrics import evaluate_run
from benchmark import load_cases, verify_lock
from day6 import ROOT, LabError, digest

INFERENCE_FILES = ['agentic.py', 'controller.py', 'run_agent.py', 'baseline.py',
                   'retrieval.py', 'day6.py', 'benchmark.py']


def code_hashes():
    return {name: digest(ROOT / 'starter' / name) for name in INFERENCE_FILES}


def resolve_config(name):
    plan = json.loads((ROOT / 'config/experiments.json').read_text(encoding='utf-8'))
    if name not in plan['experiments']:
        raise LabError('Unknown dev experiment')
    base = json.loads((ROOT / 'config/agent.json').read_text(encoding='utf-8'))
    return {**base, **plan['experiments'][name]['overrides']}


def selection_key(row):
    metrics, usage = row['metrics'], row['usage']
    citations = metrics['citation_validity']
    cost = usage['service_cost_usd']
    return (-metrics['strict_pass']['n'], -metrics['grounded_sum'],
            -(citations['n'] / citations['N'] if citations['N'] else 0),
            -metrics['context_evidence_rate']['n'], cost if cost is not None else float('inf'),
            metrics['median_ms'], row['experiment'])


def load_selection(path, variant):
    if variant not in ('baseline', 'selected'):
        raise LabError('Test variant must be baseline or selected')
    selection = json.loads(Path(path).read_text(encoding='utf-8'))
    if (selection['benchmark_sha256'] != verify_lock()
            or any(selection['code_sha256'].get(name) != value for name, value in code_hashes().items())):
        raise LabError('Frozen code or benchmark changed; do not use this selection for test')
    for name, expected in selection['proof_sha256'].items():
        if Path(name).name in ('reviews.json', 'metrics.json'):
            continue
        if digest(ROOT / name) != expected:
            raise LabError(f'Frozen dev/config evidence changed: {name}')
    experiment = 'baseline' if variant == 'baseline' else selection['selected_experiment']
    matches = [row for row in selection['dev_comparison'] if row['experiment'] == experiment]
    if len(matches) != 1:
        raise LabError('The frozen variant requires exactly one matching dev run')
    row = matches[0]
    summary_name = (Path(row['run']) / 'summary.json').as_posix()
    if summary_name not in selection['proof_sha256']:
        raise LabError('The frozen variant is missing its dev summary checksum')
    summary = json.loads((ROOT / summary_name).read_text(encoding='utf-8'))
    config = selection[f'{variant}_config']
    if (summary.get('split') != 'dev' or summary.get('experiment') != experiment
            or not (config == summary.get('config') == row.get('config') == resolve_config(experiment))):
        raise LabError('The frozen configuration differs from the proven dev run or declared experiment')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    plan_path = ROOT / 'config/experiments.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    cases = load_cases('dev')
    chunks = json.loads((ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
    rows, proof = [], {'config/agent.json': digest(ROOT / 'config/agent.json'),
                       'config/experiments.json': digest(plan_path)}
    for run in args.runs:
        run = run.resolve()
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        experiment = summary['experiment']
        if (summary['split'] != 'dev' or set(summary['case_ids']) != {c['id'] for c in cases}
                or any(summary['code_sha256'].get(name) != value for name, value in code_hashes().items())
                or summary['input_sha256'] != verify_lock('dev')
                or summary['config'] != resolve_config(experiment)
                or summary['experiments_sha256'] != digest(plan_path)):
            raise LabError('Selection requires complete, comparable dev runs with the declared configurations')
        traces = json.loads((run / 'traces.json').read_text(encoding='utf-8'))
        reviews = json.loads((run / 'reviews.json').read_text(encoding='utf-8'))
        metrics = evaluate_run(chunks, cases, traces, reviews)
        (run / 'metrics.json').write_text(json.dumps({'split': 'dev', **metrics}, ensure_ascii=False, indent=2) + '\n',
                                          encoding='utf-8', newline='\n')
        rows.append({'experiment': experiment, 'run': run.relative_to(ROOT).as_posix(), 'config': summary['config'],
                     'metrics': {k: v for k, v in metrics.items() if k != 'original_query_traces'}, 'usage': summary['usage']})
        for name in ['summary.json', 'traces.json']:
            proof[(run / name).relative_to(ROOT).as_posix()] = digest(run / name)
    if len(rows) != len(plan['experiments']) or {row['experiment'] for row in rows} != set(plan['experiments']):
        raise LabError('Include exactly one complete run for every declared dev experiment')
    chosen = min(rows, key=selection_key)
    baseline = next(row for row in rows if row['experiment'] == 'baseline')
    selection = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'selected_on': 'dev',
                 'selected_experiment': chosen['experiment'], 'selection_order': plan['selection_order'],
                 'baseline_config': baseline['config'], 'selected_config': chosen['config'],
                 'benchmark_sha256': verify_lock(), 'code_sha256': code_hashes(),
                 'proof_sha256': proof, 'dev_comparison': rows}
    output = ROOT / 'results/experiments' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    (output / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n',
                                           encoding='utf-8', newline='\n')
    print(json.dumps({'selected': chosen['experiment'], 'selection': str(output / 'selection.json')}, indent=2))


if __name__ == '__main__':
    main()
