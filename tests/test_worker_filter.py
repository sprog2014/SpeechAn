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

# Добавляем src в путь
sys.path.append('src')

class TestWorkerFilter(unittest.TestCase):
    @patch('worker.get_pg_connection')
    @patch('worker.fetch_call_metadata')
    @patch('worker.check_phone_usage')
    @patch('worker.get_default_prompt')
    @patch('worker.check_evaluation_exists')
    def test_process_file_skips_on_unallowed_numbers(self, mock_eval_exists, mock_get_prompt, mock_check_usage, mock_fetch_meta, mock_pg):
        mock_pg.return_value.__enter__.return_value = MagicMock()
        mock_eval_exists.return_value = False
        mock_get_prompt.return_value = {'id': 1, 'prompt_text': 'test'}

        # Случай 1: Оба номера не разрешены
        mock_fetch_meta.return_value = {'src': '101', 'answeredext': '102', 'direction': 'inbound'}
        mock_check_usage.side_effect = lambda num, conn=None: False

        from worker import process_file
        with patch('worker.upsert_call') as mock_upsert:
            process_file("some/path/123.mp3")
            mock_upsert.assert_not_called()

        # Случай 2: Один из номеров разрешен
        mock_check_usage.side_effect = lambda num, conn=None: num == '101'
        with patch('worker.upsert_call') as mock_upsert:
            # Нам нужно замокать остальные части чтобы не упало дальше
            with patch('worker.check_transcript_exists', return_value=True):
                # worker использует pg_conn.cursor()
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
