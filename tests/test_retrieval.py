import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from baseline import TfidfIndex, evaluate as evaluate_baseline
from retrieval import HybridRetriever, evaluate, summarize


def chunk(cid, text='refund policy', section='unit_1', score=0.9):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf',
            'page': 5, 'metadata': {'section': section}, 'score': score}


def case(cid='q', answerable=True):
    result = {'id': cid, 'split': 'dev', 'question': 'refund?', 'answerable': answerable}
    if answerable:
        result['gold'] = [{'source_id': 'doc', 'page': 5, 'evidence': 'refund policy'}]
    return result


class RetrievalTests(unittest.TestCase):
    def test_lexical_filters_before_top_k_and_baseline_records_filter(self):
        index = TfidfIndex([chunk('a', 'refund'), chunk('b', section='unit_2')])
        self.assertEqual(index.search('refund', top_k=1)[0]['id'], 'a')
        self.assertEqual(index.search('refund', top_k=1, filters={'section': 'unit_2'})[0]['id'], 'b')
        row = evaluate_baseline(index, [{**case(), 'filter': {'section': 'unit_2'}}])[0]
        self.assertEqual(row['filter'], {'section': 'unit_2'})
        self.assertEqual(row['results'][0]['id'], 'b')
        self.assertTrue(row['hit1'])

    def test_existing_metadata_filters_and_positional_top_k_still_work(self):
        row = chunk('a')
        row['metadata']['version'] = '2025'
        index = TfidfIndex([row])
        self.assertEqual(index.search('refund', 1, filters={'version': '2025'})[0]['id'], 'a')
        self.assertEqual(index.search('refund', filters={'version': 'old'}), [])
        self.assertEqual(index.search('absentterm'), [])
        for query, cutoff in [('', 3), ('q', 0), ('q', True)]:
            with self.assertRaises(ValueError):
                index.search(query, cutoff)

    def test_rrf_uses_rank_not_raw_score_and_keeps_component_trace(self):
        dense = Mock(search=Mock(return_value=[chunk('a', score=0.99), chunk('b', score=0.01)]))
        lexical = Mock(search=Mock(return_value=[chunk('b', score=1000), chunk('c', score=999)]))
        hybrid = HybridRetriever(dense, lexical, candidate_k=10, rrf_k=60)
        rows = hybrid.search('refund', filters={'section': 'unit_1'}, top_k=3)
        self.assertEqual([row['id'] for row in rows], ['b', 'a', 'c'])
        self.assertAlmostEqual(rows[0]['score'], 1 / 61 + 1 / 62)
        self.assertEqual(rows[0]['components']['dense']['rank'], 2)
        self.assertEqual(rows[0]['components']['tfidf']['rank'], 1)
        self.assertEqual(rows[0]['rank'], 1)
        for retriever in [dense, lexical]:
            retriever.search.assert_called_once_with('refund', filters={'section': 'unit_1'}, top_k=10)
        self.assertEqual([step['status'] for step in hybrid.last_steps], ['ok', 'ok'])
        self.assertEqual(len(hybrid.last_steps[0]['results']), 2)
        with self.assertRaises(ValueError):
            hybrid.search(' ')
        self.assertEqual(hybrid.last_steps, [])

    def test_rrf_breaks_ties_by_id_and_handles_empty_component(self):
        dense, lexical = Mock(), Mock()
        dense.search.return_value = [chunk('b'), chunk('a')]
        lexical.search.return_value = [chunk('a'), chunk('b')]
        hybrid = HybridRetriever(dense, lexical)
        self.assertEqual([row['id'] for row in hybrid.search('q')], ['a', 'b'])
        dense.search.return_value = []
        self.assertEqual([row['id'] for row in hybrid.search('q')], ['a', 'b'])
        lexical.search.return_value = []
        self.assertEqual(hybrid.search('q'), [])

    def test_rrf_does_not_hide_failed_component_or_count_duplicates(self):
        dense = Mock(search=Mock(return_value=[chunk('a')]))
        lexical = Mock(search=Mock(side_effect=ConnectionError('password=secret')))
        hybrid = HybridRetriever(dense, lexical)
        trace = evaluate(hybrid, [case()], top_k=3, budget=2400)[0]
        self.assertEqual(trace['status'], 'error')
        self.assertEqual([step['status'] for step in trace['steps']], ['ok', 'error'])
        self.assertNotIn('secret', str(trace))
        dense.search.return_value = [chunk('a'), chunk('a')]
        with self.assertRaises(ValueError):
            hybrid.search('q')

    def test_only_query_and_explicit_filter_reach_retriever(self):
        q = {**case(), 'filter': {'section': 'unit_1'}, 'expected_answer': 'not tool input'}
        retriever = Mock(spec=['search'], search=Mock(return_value=[chunk('a')]))
        trace = evaluate(retriever, [q], top_k=3, budget=2400)[0]
        retriever.search.assert_called_once_with('refund?', filters={'section': 'unit_1'}, top_k=3)
        self.assertNotIn('gold', trace)
        self.assertNotIn('expected_answer', trace)
        self.assertTrue(trace['hit1'] and trace['hit3'] and trace['context_hit'])

    def test_errors_stay_in_denominator_and_unanswerable_has_no_retrieval_label(self):
        cases = [case('a'), case('b'), case('c', False)]
        retriever = Mock(spec=['search'], search=Mock(side_effect=[
            [chunk('a')], ConnectionError('offline'), []]))
        traces = evaluate(retriever, cases, top_k=3, budget=2400)
        report = summarize(cases, traces)
        self.assertEqual(report['hit1'], {'n': 1, 'N': 2, 'rate': 0.5})
        self.assertEqual(report['errors'], 1)
        self.assertEqual(report['cases'], 3)
        self.assertNotIn('hit3', traces[2])

    def test_packed_context_can_miss_retrieved_evidence(self):
        retriever = Mock(spec=['search'], search=Mock(return_value=[chunk('a')]))
        trace = evaluate(retriever, [case()], top_k=3, budget=1)[0]
        self.assertTrue(trace['hit3'])
        self.assertFalse(trace['context_hit'])
        with self.assertRaises(ValueError):
            evaluate(retriever, [case()], top_k=2, budget=2400)


if __name__ == '__main__':
    unittest.main()
