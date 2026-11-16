import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
import benchmark
from day6 import digest

CHUNK = {'id': 'a', 'source_id': 'doc', 'source': 'doc.pdf', 'page': 5,
         'text': 'A commit records a snapshot.', 'metadata': {'section': 'unit_1'}}
CASE = {'id': 'D01', 'topic': 'commit', 'split': 'dev', 'question': 'What is a commit?',
        'answerable': True, 'expected_behavior': 'search_and_answer', 'expected_answer': 'A snapshot.',
        'gold': [{'source_id': 'doc', 'page': 5, 'evidence': 'A commit records a snapshot.'}]}


class BenchmarkTests(unittest.TestCase):
    def test_bad_gold_split_and_duplicates_are_rejected(self):
        self.assertEqual(benchmark.validate_cases([CHUNK], [CASE], 'dev'), {'answerable': 1, 'unanswerable': 0})
        for key, value in [('split', 'test'), ('answerable', 1), ('expected_behavior', 'answer_without_search'),
                           ('gold', [{'source_id': 'doc', 'page': 6, 'evidence': 'snapshot'}])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.validate_cases([CHUNK], [{**CASE, key: value}], 'dev')
        with self.assertRaises(ValueError):
            benchmark.validate_cases([CHUNK], [CASE, copy.deepcopy(CASE)], 'dev')

    def test_absence_requires_check_and_must_not_carry_gold(self):
        missing = {'id': 'D16', 'topic': 'limit', 'split': 'dev', 'question': 'How many?', 'answerable': False,
                   'expected_behavior': 'search_and_abstain', 'absence_reason': 'No limit given.',
                   'absence_check': {'terms': ['limit'], 'inspected_pages': [5]}}
        self.assertEqual(benchmark.validate_cases([CHUNK], [missing], 'dev')['unanswerable'], 1)
        for changes in [{'absence_check': {}}, {'absence_reason': ''}, {'gold': []}, {'expected_answer': '100'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                benchmark.validate_cases([CHUNK], [{**missing, **changes}], 'dev')
        for terms, pages in [('limit', [5]), ([''], [5]), ([1], [5]), (['limit'], '5'),
                             (['limit'], [-99]), (['limit'], [6]), (['limit'], [True])]:
            with self.subTest(terms=terms, pages=pages), self.assertRaises(ValueError):
                benchmark.validate_cases([CHUNK], [{**missing, 'absence_check': {
                    'terms': terms, 'inspected_pages': pages}}], 'dev')

    def test_case_must_be_an_object(self):
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            benchmark.validate_cases([CHUNK], [None], 'dev')

    def test_filter_must_come_from_explicit_question_scope(self):
        scoped = {**CASE, 'question': 'In Unit 1, what is a commit?', 'filter': {'section': 'unit_1'},
                  'filter_reason': 'Explicit Unit 1.'}
        benchmark.validate_cases([CHUNK], [scoped], 'dev')
        with self.assertRaises(ValueError):
            benchmark.validate_cases([CHUNK], [{**scoped, 'question': CASE['question']}], 'dev')

    def test_dev_loader_never_opens_test_file_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'benchmark').mkdir()
            values = {'data/chunks.json': [CHUNK], 'data/day6.json': {},
                      'benchmark/plan.json': {}, 'benchmark/dev.json': [CASE]}
            for name, value in values.items():
                (root / name).write_text(json.dumps(value), encoding='utf-8')
            lock = {'sha256': {name: digest(root / name) for name in values}}
            lock['sha256']['benchmark/test.json'] = 'not-a-real-checksum'
            (root / 'benchmark/lock.json').write_text(json.dumps(lock), encoding='utf-8')
            with patch.object(benchmark, 'ROOT', root):
                self.assertEqual(benchmark.load_cases('dev'), [CASE])
                (root / 'benchmark/dev.json').write_text('[]', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Frozen benchmark'):
                    benchmark.load_cases('dev')

    def test_real_benchmark_meets_minimums_without_running_retrieval(self):
        with patch('builtins.print'), patch.object(sys, 'argv', ['benchmark.py']):
            benchmark.main()

    def test_source_checked_fast_forward_alternative_is_not_a_false_negative(self):
        from baseline import hit
        case = next(c for c in benchmark.load_cases('dev') if c['id'] == 'D09')
        chunks = json.loads((benchmark.ROOT / 'data/chunks.json').read_text(encoding='utf-8'))
        panel = next(c for c in chunks if c['page'] == 31 and c['id'].startswith('54c842f61562'))
        self.assertTrue(hit(panel, case['gold']))
        self.assertTrue(any(g.get('region') for g in case['gold'] if g['page'] == 31))


if __name__ == '__main__':
    unittest.main()
