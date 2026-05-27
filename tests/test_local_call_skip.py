import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Мокаем все тяжелые зависимости ДО импорта worker
sys.modules['asr'] = MagicMock()
sys.modules['emotion'] = MagicMock()
sys.modules['llm_analysis'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['models'] = MagicMock()

sys.path.append('src')

class TestLocalCallSkip(unittest.TestCase):
    @patch('worker.get_pg_connection')
    @patch('worker.fetch_call_metadata')
    @patch('worker.is_phone_registered')
    @patch('worker.get_system_setting')
    @patch('worker.get_default_prompt')
    @patch('worker.check_phone_usage')
    @patch('worker.check_evaluation_exists')
    def test_local_call_skipped(self, mock_eval_exists, mock_usage, mock_get_prompt, mock_setting, mock_registered, mock_fetch_meta, mock_pg):
        mock_eval_exists.return_value = False
        mock_pg.return_value.__enter__.return_value = MagicMock()
        mock_get_prompt.return_value = {'id': 1, 'prompt_text': 'test'}
        mock_fetch_meta.return_value = {'src': '101', 'answeredext': '102', 'direction': 'inbound'}

        # Опция включена
        mock_setting.return_value = 'true'
        # Оба номера зарегистрированы (локальный звонок)
        mock_registered.side_effect = lambda num, conn=None: True

        from worker import process_file
        with patch('worker.upsert_call') as mock_upsert:
            process_file("some/path/123.mp3")
            mock_upsert.assert_not_called()

    @patch('worker.get_pg_connection')
    @patch('worker.fetch_call_metadata')
    @patch('worker.is_phone_registered')
    @patch('worker.get_system_setting')
    @patch('worker.get_default_prompt')
    @patch('worker.check_phone_usage')
    @patch('worker.check_evaluation_exists')
    def test_local_call_not_skipped_when_option_disabled(self, mock_eval_exists, mock_usage, mock_get_prompt, mock_setting, mock_registered, mock_fetch_meta, mock_pg):
        mock_eval_exists.return_value = False
        mock_pg.return_value.__enter__.return_value = MagicMock()
        mock_get_prompt.return_value = {'id': 1, 'prompt_text': 'test'}
        mock_fetch_meta.return_value = {'src': '101', 'answeredext': '102', 'direction': 'inbound'}

        # Опция ВЫКЛЮЧЕНА
        mock_setting.return_value = 'false'
        # Оба номера зарегистрированы
        mock_registered.side_effect = lambda num, conn=None: True
        # Но один из них разрешен для анализа
        mock_usage.side_effect = lambda num, conn=None: num == '101'

        from worker import process_file
        with patch('worker.upsert_call') as mock_upsert:
             with patch('worker.check_transcript_exists', return_value=True):
                mock_pg_conn = mock_pg.return_value.__enter__.return_value
                mock_cur = mock_pg_conn.cursor.return_value
                mock_cur.fetchall.return_value = [('operator', 'hello', 0)]

                with patch('worker.analyze_transcript', return_value={}):
                    with patch('worker.insert_evaluation'):
                        with patch('worker.set_call_done'):
                            process_file("some/path/123.mp3")
                            mock_upsert.assert_called_once()

if __name__ == '__main__':
    unittest.main()
