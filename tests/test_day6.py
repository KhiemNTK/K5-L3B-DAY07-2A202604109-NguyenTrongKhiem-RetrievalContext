"""Adapter and snapshot integrity checks; these do not measure retrieval quality."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
import day6
from convert_day6 import convert_chunk

ROW = {'chunk_id': 'c1', 'document_id': 'doc', 'source': 'doc.pdf', 'page': 5,
       'text': 'A Git commit records a snapshot.', 'metadata': {'section': 'unit_1'},
       'score': 0.9, 'embedding': '[1,0]'}


class Day6Tests(unittest.TestCase):
    def setUp(self):
        self.embedder = SimpleNamespace(dimensions=2, embed_query=Mock(return_value=[1, 0]))
        self.store = SimpleNamespace(dimensions=2, search=Mock(return_value=[copy.deepcopy(ROW)]))
        self.retriever = day6.Day6Retriever(self.embedder, self.store)

    def test_query_embedding_and_source_are_preserved(self):
        result = self.retriever.search('What is a commit?', top_k=7)
        self.embedder.embed_query.assert_called_once_with('What is a commit?')
        self.store.search.assert_called_once_with('B', [1, 0], top_k=7, filters=None)
        self.assertEqual(result, [{**convert_chunk(ROW), 'score': 0.9, 'rank': 1}])

    def test_explicit_section_uses_the_same_B_index_via_C(self):
        self.retriever.search('What is a commit?', filters={'section': 'unit_1'})
        self.store.search.assert_called_once_with('C', [1, 0], top_k=3, filters={'section': 'unit_1'})

    def test_invalid_inputs_fail_before_embedding_or_database(self):
        for query, top_k, filters in [('', 3, None), (None, 3, None), ('q', True, None),
                                       ('q', 0, None), ('q', 3, {'gold_page': 5}),
                                       ('q', 3, {'section': ''}), ('q', 3, [])]:
            with self.subTest(query=query, top_k=top_k, filters=filters), self.assertRaises(ValueError):
                self.retriever.search(query, top_k=top_k, filters=filters)
        self.embedder.embed_query.assert_not_called()
        self.store.search.assert_not_called()

    def test_no_fake_fallback_on_empty_results_or_backend_errors(self):
        self.store.search.return_value = []
        self.assertEqual(self.retriever.search('q'), [])
        self.store.search.side_effect = ConnectionError('offline')
        with self.assertRaises(ConnectionError):
            self.retriever.search('q')

    def test_dimension_mismatch_is_rejected(self):
        self.store.dimensions = 3
        with self.assertRaises(ValueError):
            day6.Day6Retriever(self.embedder, self.store)

    def test_verify_index_checks_content_and_vectors_not_only_counts(self):
        chunks = [convert_chunk(ROW)]
        day6.verify_index([ROW], chunks, [[1, 0]], ['c1'], 2)
        for changes in [{'text': 'Changed content'}, {'embedding': '[0,1]'},
                        {'embedding': '[NaN,0]'}, {'chunk_id': 'old'}, {'page': 6},
                        {'metadata': {'section': 'unit_2'}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                day6.verify_index([{**ROW, **changes}], chunks, [[1, 0]], ['c1'], 2)
        for vectors, ids, dimensions in [([[1, 0]], ['wrong'], 2), ([[1, 0]], ['c1'], 3),
                                         ([[2, 0]], ['c1'], 2), ([], [], 2)]:
            with self.subTest(ids=ids, dimensions=dimensions), self.assertRaises(ValueError):
                day6.verify_index([ROW], chunks, vectors, ids, dimensions)

    def test_changed_source_or_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'results').mkdir()
            original = root / 'results/chunks_B.jsonl'
            original.write_text(json.dumps(ROW) + '\n', encoding='utf-8')
            exported = root / 'data/chunks.json'
            exported.write_text(json.dumps([convert_chunk(ROW)]), encoding='utf-8')
            contract = {'source_sha256': {'results/chunks_B.jsonl': day6.digest(original)},
                        'corpus': {'chunk_count': 1}}
            (root / 'data/day6.json').write_text(json.dumps(contract), encoding='utf-8')
            with patch.object(day6, 'ROOT', root):
                self.assertEqual(day6.load_snapshot(root)[1], [convert_chunk(ROW)])
                exported.write_text('[]', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Day 7 chunks'):
                    day6.load_snapshot(root)
                original.write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Day 6 snapshot changed'):
                    day6.load_snapshot(root)

    def test_cli_records_failure_without_leaking_connection_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(day6, 'ROOT', root), patch.object(sys, 'argv', [
                    'day6.py', '--day6-root', directory, '--query', 'q']), patch.object(
                    day6, 'connect_day6', side_effect=ConnectionError('password=secret')), patch('builtins.print'):
                with self.assertRaises(SystemExit) as failure:
                    day6.main()
            self.assertEqual(failure.exception.code, 1)
            files = list((root / 'results/day6').glob('*/trace.json'))
            self.assertEqual(len(files), 1)
            text = files[0].read_text(encoding='utf-8')
            self.assertNotIn('secret', text)
            trace = json.loads(text)
            self.assertEqual(trace['status'], 'error')
            self.assertEqual(trace['phase'], 'connect_and_verify')
            self.assertEqual(trace['error_type'], 'ConnectionError')
            self.assertTrue(trace['message'])

    def test_safe_setup_errors_are_distinct_and_driver_messages_are_hidden(self):
        missing = day6.error_details(day6.LabError('DATABASE_URL is missing; configure .env'))
        changed = day6.error_details(day6.LabError('Day 6 snapshot changed: chunks_B.jsonl'))
        self.assertNotEqual(missing['message'], changed['message'])
        for error in [ValueError('password=secret'), ConnectionError('postgres://user:secret@db')]:
            self.assertNotIn('secret', str(day6.error_details(error)))

    def test_numeric_embedding_digest_ignores_python_number_type_but_detects_changes(self):
        self.assertEqual(day6.embedding_digest([[1, 0]]), day6.embedding_digest([[1.0, 0.0]]))
        self.assertNotEqual(day6.embedding_digest([[1, 0]]), day6.embedding_digest([[0, 1]]))

    def test_telemetry_and_historical_run_files_do_not_block_snapshot_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'results').mkdir()
            original = root / 'results/chunks_B.jsonl'
            original.write_text(json.dumps(ROW) + '\n', encoding='utf-8')
            (root / 'data/chunks.json').write_text(json.dumps([convert_chunk(ROW)]), encoding='utf-8')
            hashes = {'results/chunks_B.jsonl': day6.digest(original),
                      'results/embedding_report.json': 'old timing', 'results/embeddings.npz': 'old ZIP metadata',
                      'results/runs/old/summary.json': 'old duration'}
            contract = {'source_sha256': hashes, 'corpus': {'chunk_count': 1}}
            (root / 'data/day6.json').write_text(json.dumps(contract), encoding='utf-8')
            with patch.object(day6, 'ROOT', root):
                self.assertEqual(day6.load_snapshot(root)[1], [convert_chunk(ROW)])


if __name__ == '__main__':
    unittest.main()
