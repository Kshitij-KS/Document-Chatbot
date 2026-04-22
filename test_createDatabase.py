import sys
from unittest.mock import MagicMock

# Mocking dependencies that are not installed in the environment to allow importing createDatabase
# This is necessary because the environment lacks the required libraries and has no internet access.
def setup_mocks():
    mock_langchain_community = MagicMock()
    mock_langchain = MagicMock()
    mock_langchain_huggingface = MagicMock()
    mock_dotenv = MagicMock()

    sys.modules["langchain_community"] = mock_langchain_community
    sys.modules["langchain_community.document_loaders"] = mock_langchain_community.document_loaders
    sys.modules["langchain_community.vectorstores"] = mock_langchain_community.vectorstores
    sys.modules["langchain"] = mock_langchain
    sys.modules["langchain.text_splitter"] = mock_langchain.text_splitter
    sys.modules["langchain.schema"] = mock_langchain.schema
    sys.modules["langchain_huggingface"] = mock_langchain_huggingface
    sys.modules["dotenv"] = mock_dotenv

setup_mocks()

import unittest
from unittest.mock import patch
import createDatabase

class TestCreateDatabase(unittest.TestCase):

    @patch('createDatabase.os.path.exists')
    @patch('createDatabase.os.makedirs')
    @patch('createDatabase.DirectoryLoader')
    def test_load_documents_empty_directory_handling(self, mock_loader_class, mock_makedirs, mock_exists):
        """
        Test that load_documents correctly handles the case where the data directory does not exist.
        It should create the directory and return an empty list.
        """
        # Scenario: DATA_PATH does not exist
        mock_exists.return_value = False

        # Call the function
        result = createDatabase.load_documents()

        # Assertions
        mock_exists.assert_called_once_with(createDatabase.DATA_PATH)
        mock_makedirs.assert_called_once_with(createDatabase.DATA_PATH, exist_ok=True)
        self.assertEqual(result, [])
        mock_loader_class.assert_not_called()

if __name__ == '__main__':
    unittest.main()
