import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Добавляем src в путь
sys.path.append('src')

class TestDbUtils(unittest.TestCase):
    @patch('db_utils.get_pg_connection')
    @patch('db_utils.get_mysql_connection')
    def test_sync_phones(self, mock_mysql, mock_pg):
        # Мокаем MySQL
        mock_mysql_conn = MagicMock()
        mock_mysql.return_value.__enter__.return_value = mock_mysql_conn
        mock_mysql_cursor = mock_mysql_conn.cursor.return_value
        mock_mysql_cursor.fetchall.return_value = [
            {'name': 'User 1', 'number': '101'},
            {'name': 'User 2', 'number': '102'}
        ]

        # Мокаем PostgreSQL (get_all_phones)
        mock_pg_conn = MagicMock()
        mock_pg.return_value.__enter__.return_value = mock_pg_conn
        mock_pg_cursor = mock_pg_conn.cursor.return_value

        # Сначала get_all_phones вызывается внутри sync
        # Он использует RealDictCursor, поэтому нам нужно это учесть или замокать get_all_phones отдельно

        with patch('db_utils.get_all_phones') as mock_get_all:
            mock_get_all.return_value = [{'number': '101', 'name': 'Old User 1', 'use': False}]

            from db_utils import sync_phones_from_external_db
            sync_phones_from_external_db()

            # Проверяем вызовы execute в PostgreSQL
            calls = mock_pg_cursor.execute.call_args_list
            # 1. DELETE FROM phones
            # 2. INSERT 101 (use=False)
            # 3. INSERT 102 (use=True)
            self.assertEqual(calls[0][0][0], "DELETE FROM phones")

            # Проверяем что 101 вставился с use=False (сохранено)
            found_101 = False
            found_102 = False
            for c in calls:
                if "INSERT INTO phones" in c[0][0]:
                    args = c[0][1]
                    if args[0] == '101':
                        self.assertEqual(args[2], False)
                        found_101 = True
                    if args[0] == '102':
                        self.assertEqual(args[2], True)
                        found_102 = True

            self.assertTrue(found_101)
            self.assertTrue(found_102)

if __name__ == '__main__':
    unittest.main()
