import copy
import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agent_metrics import evaluate_run
from benchmark import load_cases, verify_lock
from day6 import LabError, digest
from experiments import code_hashes, load_selection, resolve_config, selection_key
from run_agent import run_config


class ExperimentTests(unittest.TestCase):
    def test_only_declared_variable_changes(self):
        baseline = resolve_config('baseline')
        self.assertEqual(resolve_config('compact_context'), {**baseline, 'context_budget_chars': 800})
        self.assertEqual(resolve_config('two_searches'), {**baseline, 'max_searches': 2})
        with self.assertRaises(LabError):
            resolve_config('unknown')

    def test_quality_precedes_cost(self):
        good = {'experiment': 'a', 'metrics': {'strict_pass': {'n': 16}, 'grounded_sum': 30,
                'citation_validity': {'n': 15, 'N': 16}, 'context_evidence_rate': {'n': 15},
                'median_ms': 100}, 'usage': {'service_cost_usd': 1}}
        cheap = copy.deepcopy(good)
        cheap['metrics']['strict_pass']['n'] = 15
        cheap['usage']['service_cost_usd'] = 0
        self.assertLess(selection_key(good), selection_key(cheap))
        cheap['metrics']['strict_pass']['n'] = 16
        self.assertLess(selection_key(cheap), selection_key(good))
        cheap['usage']['service_cost_usd'] = None
        self.assertLess(selection_key(good), selection_key(cheap))

    def test_test_disallows_subsets_and_config_overrides(self):
        args = Namespace(split='test', selection=Path('selection.json'), variant='selected',
                         case=None, method=None, experiment=None)
        with patch('run_agent.load_selection', return_value={'frozen': True}) as load:
            self.assertEqual(run_config(args), {'frozen': True})
            load.assert_called_once_with(args.selection, 'selected')
        for name, value in [('case', ['T01']), ('method', 'dense'), ('experiment', 'baseline'),
                            ('selection', None), ('variant', None)]:
            bad = copy.copy(args)
            setattr(bad, name, value)
            with self.assertRaises(LabError):
                run_config(bad)

    def test_frozen_code_inputs_and_dev_proof_are_verified(self):
        config = {'max_searches': 3}
        selection = {'benchmark_sha256': {'data': 'a'}, 'code_sha256': {'code': 'b'},
                     'proof_sha256': {'results/dev/summary.json': 'c'},
                     'selected_experiment': 'baseline', 'selected_config': config,
                     'dev_comparison': [{'experiment': 'baseline', 'run': 'results/dev', 'config': config}]}
        summary = {'split': 'dev', 'experiment': 'baseline', 'config': config}
        with patch('experiments.Path.read_text', autospec=True, side_effect=lambda path, **kw:
                   json.dumps(selection if path.name == 'selection.json' else summary)), \
             patch('experiments.resolve_config', return_value=config), \
             patch('experiments.verify_lock', return_value={'data': 'a'}) as inputs, \
             patch('experiments.code_hashes', return_value={'code': 'b'}) as code, \
             patch('experiments.digest', return_value='c') as proof:
            self.assertEqual(load_selection('selection.json', 'selected'), config)
            selection['proof_sha256']['results/dev/reviews.json'] = 'previous-review'
            self.assertEqual(load_selection('selection.json', 'selected'), config)
            proof.return_value = 'changed'
            with self.assertRaises(LabError): load_selection('selection.json', 'selected')
            proof.return_value = 'c'
            code.return_value = {'code': 'changed'}
            with self.assertRaises(LabError): load_selection('selection.json', 'selected')
            code.return_value = {'code': 'b'}
            inputs.return_value = {'data': 'changed'}
            with self.assertRaises(LabError): load_selection('selection.json', 'selected')

    def test_selection_config_must_match_proven_dev_config(self):
        path = Path(__file__).resolve().parents[1] / 'results/experiments/20261116T043650385003Z/selection.json'
        frozen = json.loads(path.read_text(encoding='utf-8'))
        read_text = Path.read_text
        self.assertEqual(load_selection(path, 'selected'), frozen['selected_config'])
        for variant in ['baseline', 'selected']:
            changed = copy.deepcopy(frozen)
            changed[f'{variant}_config']['max_searches'] = 99
            with self.subTest(variant=variant), \
                 patch('experiments.Path.read_text', autospec=True, side_effect=lambda p, **kw:
                       json.dumps(changed) if p.resolve() == path else read_text(p, **kw)):
                with self.assertRaises(LabError):
                    load_selection(path, variant)

    def test_selection_requires_a_proven_dev_summary(self):
        path = Path(__file__).resolve().parents[1] / 'results/experiments/20261116T043650385003Z/selection.json'
        changed = json.loads(path.read_text(encoding='utf-8'))
        changed['proof_sha256'] = {}
        read_text = Path.read_text
        with patch('experiments.Path.read_text', autospec=True, side_effect=lambda p, **kw:
                   json.dumps(changed) if p.resolve() == path else read_text(p, **kw)):
            with self.assertRaises(LabError):
                load_selection(path, 'selected')

    def test_full_denominators_and_context_validation(self):
        chunk = {'id': 'c1', 'text': 'Alpha records history.', 'source_id': 'book',
                 'source': 'book.pdf', 'page': 1, 'metadata': {}}
        case = {'id': 'a', 'question': 'Alpha history', 'answerable': True,
                'gold': [{'source_id': 'book', 'page': 1, 'evidence': 'records history'}]}
        cases = [case, {**case, 'id': 'b'}]
        trace = {'query_id': 'a', 'status': 'answered', 'method': 'tfidf', 'elapsed_ms': 10,
                 'events': [{'event': 'search'}], 'citations': ['c1'], 'context_ids': ['c1']}
        traces = [trace, {**trace, 'query_id': 'b', 'status': 'error'}]
        reviews = [{'query_id': 'a', 'reason': 'Page 1 directly supports the answer.',
                    'answer_grade': 2, 'context_grade': 2, 'citation_support': [True]},
                   {'query_id': 'b', 'context_grade': 2,
                    'reason': 'Complete context was retrieved before the later model request failed.'}]
        result = evaluate_run([chunk], cases, traces, reviews)
        self.assertEqual(result['strict_pass']['n'], 1)
        self.assertEqual(result['strict_pass']['N'], 2)
        self.assertEqual(result['context_evidence_rate']['N'], 2)
        self.assertEqual(result['context_evidence_rate']['n'], 2)
        self.assertEqual(result['context_grades'], {'0': 0, '1': 0, '2': 2})
        self.assertEqual(result['grounded_mean_including_errors'], 1)
        self.assertEqual(result['hit3']['n'], 2)
        with self.assertRaises(ValueError): evaluate_run([chunk], cases, traces, reviews[:1])
        trace['context_ids'] = []
        with self.assertRaises(ValueError): evaluate_run([chunk], cases, traces, reviews)

    def test_error_context_needs_a_review_when_evidence_exists(self):
        chunk = {'id': 'c1', 'text': 'Alpha records history.', 'source_id': 'book',
                 'source': 'book.pdf', 'page': 1, 'metadata': {}}
        case = {'id': 'a', 'question': 'Alpha history', 'answerable': True,
                'gold': [{'source_id': 'book', 'page': 1, 'evidence': 'records history'}]}
        trace = {'query_id': 'a', 'status': 'error', 'method': 'tfidf', 'elapsed_ms': 10,
                 'events': [], 'context_ids': ['c1']}
        review = {'query_id': 'a', 'reason': 'The model request failed after retrieval.'}
        with self.assertRaises(ValueError):
            evaluate_run([chunk], [case], [trace], [review])
        trace['context_ids'] = []
        metrics = evaluate_run([chunk], [case], [trace], [review])
        self.assertEqual(metrics['context_grades'], {'0': 1, '1': 0, '2': 0})

    def test_saved_results_match_reviews_traces_and_selected_pipeline(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'results/experiments/20261116T043650385003Z/selection.json'
        def load(path):
            return json.loads(path.read_text(encoding='utf-8'))
        frozen = load(path)
        load_selection(path, 'selected')
        chunks = load(root / 'data/chunks.json')
        runs = [root / row['run'] for row in frozen['dev_comparison']]
        runs += [p.parent for p in (root / 'results/agent').glob('*-test/summary.json')]
        self.assertEqual(len(runs), 5)
        rows = []
        for run in runs:
            summary = load(run / 'summary.json')
            self.assertEqual(summary['input_sha256'], verify_lock(summary['split']))
            for name, expected in code_hashes().items():
                self.assertEqual(summary['code_sha256'][name], expected)
            reviews = load(run / 'reviews.json')
            metrics = evaluate_run(chunks, load_cases(summary['split']),
                                   load(run / 'traces.json'), reviews)
            actual = json.loads(json.dumps({'split': summary['split'], **metrics}))
            self.assertEqual(actual, load(run / 'metrics.json'))
            if summary['split'] == 'test':
                self.assertEqual(summary['selection_sha256'], digest(path))
                self.assertEqual(summary['config'], frozen[f"{summary['variant']}_config"])
            else:
                rows.append({'experiment': summary['experiment'], 'metrics': metrics, 'usage': summary['usage']})
        winner = min(rows, key=selection_key)['experiment']
        self.assertEqual(winner, frozen['selected_experiment'])


if __name__ == '__main__':
    unittest.main()
