"""Validate benchmark labels without sending them to a retriever or model."""
import argparse
import json
import re
from pathlib import Path

from baseline import validate, validate_search
from day6 import ROOT, digest


def validate_cases(chunks, cases, split):
    if not isinstance(cases, list) or not cases:
        raise ValueError('Cases must be a non-empty JSON array')
    ids, topics, questions = set(), set(), set()
    for case in cases:
        for key, seen in [('id', ids), ('topic', topics), ('question', questions)]:
            value = case.get(key)
            normalized = ' '.join(value.casefold().split()) if isinstance(value, str) else ''
            if not normalized or normalized in seen:
                raise ValueError(f'Empty or duplicate case {key}')
            seen.add(normalized)
        if case.get('split') != split or type(case.get('answerable')) is not bool:
            raise ValueError('Every case needs the correct split and answerable boolean')
        filters = case.get('filter')
        validate_search(case['question'], 3, filters)
        if filters:
            section = filters.get('section', '')
            if (set(filters) != {'section'} or not re.fullmatch(r'unit_[1-5]', section)
                    or not re.search(r'\bUnit\s+' + section[-1] + r'\b', case['question'], re.I)
                    or not case.get('filter_reason')):
                raise ValueError('A section filter must be explicit in the question and explained')
        if case['answerable']:
            if case.get('expected_behavior') != 'search_and_answer':
                raise ValueError('Answerable cases must require search_and_answer')
        elif (case.get('expected_behavior') != 'search_and_abstain' or not case.get('absence_reason')
              or not isinstance(case.get('absence_check'), dict) or not case['absence_check'].get('terms')
              or not case['absence_check'].get('inspected_pages') or 'gold' in case or 'expected_answer' in case):
            raise ValueError('Unanswerable cases need an absence check and must not contain gold answers')
    answerable = [case for case in cases if case['answerable']]
    if answerable:
        validate(chunks, answerable)
    return {'answerable': len(answerable), 'unanswerable': len(cases) - len(answerable)}


def verify_lock(split=None):
    lock = json.loads((ROOT / 'benchmark/lock.json').read_text(encoding='utf-8'))
    names = ['data/chunks.json', 'data/day6.json', 'benchmark/plan.json']
    names += [f'benchmark/{name}.json' for name in ([split] if split else ['dev', 'test'])]
    for name in names:
        if digest(ROOT / name) != lock['sha256'][name]:
            raise ValueError(f'Frozen benchmark input changed: {name}')
    return {name: lock['sha256'][name] for name in names}


def load_cases(split):
    verify_lock(split)
    cases = json.loads((ROOT / f'benchmark/{split}.json').read_text(encoding='utf-8'))
    chunks = json.loads((ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
    validate_cases(chunks, cases, split)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify_lock()
    chunks = json.loads((ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
    counts, cases = {}, []
    for split, minimum in [('dev', (15, 2)), ('test', (5, 2))]:
        selected = load_cases(split)
        counts[split] = validate_cases(chunks, selected, split)
        if counts[split]['answerable'] < minimum[0] or counts[split]['unanswerable'] < minimum[1]:
            raise ValueError(f'{split}: fewer than the required cases')
        cases.extend(selected)
    for key in ['id', 'topic', 'question']:
        values = [' '.join(c[key].casefold().split()) for c in cases]
        if len(values) != len(set(values)):
            raise ValueError(f'Dev/test overlap: {key}')
    print(json.dumps({'status': 'valid', 'counts': counts, 'retrieval_executed': False}, indent=2))


if __name__ == '__main__':
    main()
