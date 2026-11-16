import copy
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from controller import OpenRouterController
from day6 import LabError
from run_agent import usage_totals, validate_config

TOOL = {'id': 'call-1', 'type': 'function', 'function': {'name': 'search', 'arguments': '{"query":"git init"}'}}
HISTORY = [{'tool': 'search', 'queries': ['git init'], 'context': '[c1] evidence', 'context_ids': ['c1']}]


def controller():
    return OpenRouterController('private-test-key', model='openai/gpt-4o-mini', max_searches=3, max_tokens=700, timeout=45)


def response(*, calls=None, content=None, finish=None):
    message = {'role': 'assistant', 'content': content}
    if calls is not None:
        message['tool_calls'] = calls
    data = {'id': 'generation-id', 'model': 'model-version',
            'choices': [{'message': message, 'finish_reason': finish or ('tool_calls' if calls else 'stop')}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120, 'cost': 0.0001}}
    return io.BytesIO(json.dumps(data).encode())


class ControllerTests(unittest.TestCase):
    def test_native_tool_call_id_and_budgeted_context_are_preserved(self):
        agent = controller()
        final = {'action': 'answer', 'answer': 'A repository [c1]', 'citations': ['c1']}
        with patch('controller.urllib.request.urlopen', side_effect=[response(calls=[TOOL]),
                    response(content=json.dumps(final))]) as request:
            self.assertEqual(agent.decide('What is git init?', [])['action'], 'search')
            self.assertEqual(agent.decide('What is git init?', HISTORY), final)
        first = json.loads(request.call_args_list[0].args[0].data)
        second = json.loads(request.call_args_list[1].args[0].data)
        self.assertEqual(first['tool_choice'], 'required')
        self.assertFalse(first['parallel_tool_calls'])
        self.assertEqual(second['tool_choice'], 'auto')
        self.assertEqual(second['messages'][-1]['tool_call_id'], 'call-1')
        self.assertEqual(json.loads(second['messages'][-1]['content']), HISTORY[0])
        self.assertEqual(len(second['messages']), 4)
        self.assertNotIn('private-test-key', json.dumps(agent.calls))
        self.assertEqual(usage_totals(agent.calls)['service_cost_usd'], 0.0002)

    def test_search_limit_disables_tools_and_allows_explicit_abstention(self):
        agent = controller()
        history = [{**HISTORY[0], 'queries': ['one', 'two', 'three']}]
        with patch('controller.urllib.request.urlopen', side_effect=[response(calls=[TOOL]),
                    response(content='{"action":"abstain"}')]) as request:
            agent.decide('q', [])
            self.assertEqual(agent.decide('q', history)['action'], 'abstain')
        self.assertEqual(json.loads(request.call_args.args[0].data)['tool_choice'], 'none')

    def test_client_renders_only_ids_selected_by_model_and_present_in_context(self):
        for cid, expected in [('c1', 'A snapshot. [c1]'), ('invented', 'A snapshot.')]:
            agent = controller()
            raw = {'action': 'answer', 'answer': 'A snapshot.', 'citations': [cid]}
            with patch('controller.urllib.request.urlopen', side_effect=[response(calls=[TOOL]),
                    response(content=json.dumps(raw))]):
                agent.decide('q', [])
                result = agent.decide('q', HISTORY)
            self.assertEqual(result['answer'], expected)
            self.assertEqual(json.loads(agent.calls[-1]['response_message']['content']), raw)

    def test_bad_tool_arguments_duplicate_ids_and_unknown_tools_fail(self):
        malformed = copy.deepcopy(TOOL)
        malformed['function']['arguments'] = 'not JSON'
        unknown = copy.deepcopy(TOOL)
        unknown['function']['name'] = 'delete'
        extra = copy.deepcopy(TOOL)
        extra['function']['arguments'] = '{"query":"q","gold_page":5}'
        for calls in [[malformed], [unknown], [extra], [TOOL, TOOL]]:
            with self.subTest(calls=calls), patch('controller.urllib.request.urlopen', return_value=response(calls=calls)):
                agent = controller()
                with self.assertRaises(LabError):
                    agent.decide('q', [])
                self.assertEqual(agent.calls[0]['status'], 'error')
                self.assertIn('response_message', agent.calls[0])

    def test_multiple_native_calls_are_executed_before_next_model_request(self):
        agent = controller()
        second = copy.deepcopy(TOOL)
        second['id'] = 'call-2'
        second['function']['arguments'] = '{"query":"restore earlier version"}'
        with patch('controller.urllib.request.urlopen', side_effect=[response(calls=[TOOL, second]),
                response(content='{"action":"abstain"}')]) as request:
            self.assertEqual(agent.decide('q', [])['tool_call_id'], 'call-1')
            self.assertEqual(agent.decide('q', HISTORY)['tool_call_id'], 'call-2')
            history = [{**HISTORY[0], 'queries': ['git init', 'restore earlier version']}]
            self.assertEqual(agent.decide('q', history)['action'], 'abstain')
        self.assertEqual(request.call_count, 2)
        messages = json.loads(request.call_args.args[0].data)['messages']
        results = [m for m in messages if m['role'] == 'tool']
        self.assertEqual([r['tool_call_id'] for r in results], ['call-1', 'call-2'])
        self.assertNotIn('context', json.loads(results[0]['content']))
        self.assertEqual(json.loads(results[-1]['content'])['context'], HISTORY[0]['context'])

    def test_truncation_and_answer_before_search_are_not_successes(self):
        for reply in [response(calls=[TOOL], finish='length'), response(content='{"action":"abstain"}')]:
            with patch('controller.urllib.request.urlopen', return_value=reply):
                with self.assertRaises(LabError):
                    controller().decide('q', [])

    def test_http_and_network_errors_do_not_log_credentials(self):
        for error in [urllib.error.HTTPError('https://host/private-test-key', 401, 'private-test-key', {}, None),
                      urllib.error.URLError('private-test-key')]:
            agent = controller()
            with patch('controller.urllib.request.urlopen', side_effect=error), self.assertRaises(LabError):
                agent.decide('q', [])
            self.assertNotIn('private-test-key', json.dumps(agent.calls))

    def test_missing_key_fails_without_a_request(self):
        with self.assertRaisesRegex(LabError, 'API_KEY is missing'):
            OpenRouterController('', model='m', max_searches=3, max_tokens=700, timeout=45)

    def test_missing_cost_is_unknown_and_known_cost_is_kept(self):
        result = usage_totals([{'usage': {'total_tokens': 10, 'cost': 0.001}}, {'usage': {}}])
        self.assertIsNone(result['service_cost_usd'])
        self.assertIsNone(result['total_tokens'])
        self.assertEqual(result['reported_cost_usd'], 0.001)

    def test_config_requires_positive_integer_limits(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'config/agent.json').read_text())
        validate_config(config)
        for changes in [{'max_searches': True}, {'timeout_seconds': 0}, {'retrieval': 'unknown'}]:
            with self.assertRaises(LabError):
                validate_config({**config, **changes})


if __name__ == '__main__':
    unittest.main()
