import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Мокаем тяжелые зависимости
sys.modules['asr'] = MagicMock()
sys.modules['emotion'] = MagicMock()
sys.modules['llm_analysis'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['models'] = MagicMock()

sys.path.append('src')

class TestPipeline(unittest.TestCase):
    @patch('worker.get_pg_connection')
    @patch('worker.fetch_call_metadata')
    @patch('worker.check_phone_usage')
    @patch('worker.get_default_prompt')
    @patch('worker.check_evaluation_exists')
    @patch('worker.check_transcript_exists')
    @patch('worker.os.path.exists')
    def test_full_pipeline_executed_when_allowed(self, mock_exists, mock_trans_exists, mock_eval_exists, mock_get_prompt, mock_check_usage, mock_fetch_meta, mock_pg):
        mock_exists.return_value = True # Pretend file exists
        mock_pg_conn = MagicMock()
        mock_pg.return_value.__enter__.return_value = mock_pg_conn
        mock_eval_exists.return_value = False
        mock_trans_exists.return_value = False # Force ASR path
        mock_get_prompt.return_value = {'id': 1, 'prompt_text': 'test'}

        mock_fetch_meta.return_value = {'src': '101', 'answeredext': '102', 'direction': 'inbound', 'calldate': '2023-01-01', 'duration': 10, 'billsec': 10, 'fromtrunksrc': '', 'moduleparams': '', 'incomingTrunk': '', 'linkedid': '123'}
        mock_check_usage.return_value = True # Allowed

        from worker import process_file

        with patch('worker.upsert_call') as mock_upsert,              patch('worker.librosa.load', return_value=(MagicMock(), 16000)),              patch('worker.sf.write'),              patch('worker.transcribe_audio', return_value=[(0, 1, 'hello')]),              patch('worker.predict_emotion', return_value=('neutral', 1.0)),              patch('worker.insert_transcript', return_value=1),              patch('worker.insert_emotion'),              patch('worker.analyze_transcript', return_value={'politeness_score': 10}),              patch('worker.insert_evaluation') as mock_ins_eval,              patch('worker.set_call_done') as mock_done,              patch('worker.set_processing_duration'):

            with patch('worker.tempfile.TemporaryDirectory') as mock_tmp:
                mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
                process_file("some/path/123.mp3")

                mock_upsert.assert_called_once()
                mock_ins_eval.assert_called_once()
                mock_done.assert_called_once()

if __name__ == '__main__':
    unittest.main()
