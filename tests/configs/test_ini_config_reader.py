import configparser
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.config.ini_config_reader import INIConfigReader


class TestINIConfigReader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.valid_ini_content = """
[database]
host = localhost
port = 5432
username = admin
enabled = true
max_connections = 100
timeout = 30.5

[api]
url = https://api.example.com
timeout = 60
retry_attempts = 3
allowed_methods = [GET, POST, PUT]
allowed_origins = (localhost, 127.0.0.1)
feature_flags = {true, false, true}

[logging]
level = INFO
file_path = /var/log/app.log
max_size = 1048576
"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.valid_config_path = Path(self.temp_dir) / 'config.ini'
        with open(self.valid_config_path, 'w') as f:
            f.write(self.valid_ini_content)

    def tearDown(self):
        """Clean up test fixtures after each test method"""
        # Remove temporary directory and its contents
        for file in Path(self.temp_dir).glob('*'):
            file.unlink()
        os.rmdir(self.temp_dir)

    def test_valid_ini_file_loading(self):
        """Test loading a valid INI file"""
        config_reader = INIConfigReader(self.valid_config_path)

        # Test if sections are loaded correctly
        self.assertIn('database', config_reader.config_data)
        self.assertIn('api', config_reader.config_data)
        self.assertIn('logging', config_reader.config_data)

        # Test if parameters are loaded correctly
        self.assertEqual(config_reader.get_param('database.host'), 'localhost')
        self.assertEqual(config_reader.get_param('database.port'), '5432')

    def test_file_not_found(self):
        """Test behavior when INI file doesn't exist"""
        non_existent_path = Path(self.temp_dir) / 'non_existent.ini'
        with self.assertRaises(FileNotFoundError):
            INIConfigReader(non_existent_path)

    def test_invalid_file_extension(self):
        """Test behavior with non-INI file"""
        invalid_path = Path(self.temp_dir) / 'config.txt'
        with open(invalid_path, 'w') as f:
            f.write(self.valid_ini_content)

        with self.assertRaises(ValueError):
            INIConfigReader(invalid_path)

    def test_empty_ini_file(self):
        """Test behavior with empty INI file"""
        empty_config_path = Path(self.temp_dir) / 'empty.ini'
        with open(empty_config_path, 'w') as f:
            f.write('')

        config_reader = INIConfigReader(empty_config_path)
        self.assertEqual(config_reader.config_data, {})

    def test_type_conversion(self):
        """Test type conversion of INI values"""
        config_reader = INIConfigReader(self.valid_config_path)

        # Test integer conversion
        self.assertEqual(
            config_reader.get_param('database.port', v_type=int),
            5432
        )

        # Test float conversion
        self.assertEqual(
            config_reader.get_param('database.timeout', v_type=float),
            30.5
        )

        # Test boolean conversion
        self.assertTrue(
            config_reader.get_param('database.enabled', v_type=bool)
        )

    def test_collection_handling(self):
        """Test handling of collection values in INI file"""
        config_reader = INIConfigReader(self.valid_config_path)

        # Test list
        methods = config_reader.get_collection('api.allowed_methods', collection_type=list)
        self.assertEqual(methods, ['GET', 'POST', 'PUT'])

        # Test tuple
        origins = config_reader.get_collection('api.allowed_origins')
        self.assertEqual(origins, ('localhost', '127.0.0.1'))

        # Test set with type conversion
        flags = config_reader.get_collection('api.feature_flags', v_type=bool, collection_type=set)
        self.assertEqual(flags, {True, False})

    def test_section_access(self):
        """Test accessing entire sections"""
        config_reader = INIConfigReader(self.valid_config_path)

        # Test getting entire section
        db_section = config_reader.get_section('database')
        self.assertIsInstance(db_section, dict)
        self.assertEqual(db_section['host'], 'localhost')

        # Test dictionary-style access
        api_section = config_reader['api']
        self.assertEqual(api_section['url'], 'https://api.example.com')

    def test_invalid_section_access(self):
        """Test accessing non-existent sections"""
        config_reader = INIConfigReader(self.valid_config_path)

        # Test get_section with invalid section
        self.assertIsNone(config_reader.get_section('invalid_section'))

        # Test dictionary access with invalid section
        with self.assertRaises(KeyError):
            _ = config_reader['invalid_section']

    @patch('configparser.ConfigParser.read')
    def test_parser_error(self, mock_read):
        """Test behavior when ConfigParser encounters an error"""
        mock_read.side_effect = configparser.Error('Parsing error')

        with self.assertRaises(configparser.Error):
            INIConfigReader(self.valid_config_path)

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects"""
        # Test with string path
        config_reader = INIConfigReader(str(self.valid_config_path))
        self.assertIsInstance(config_reader.config_path, Path)

        # Test with Path object
        config_reader = INIConfigReader(self.valid_config_path)
        self.assertIsInstance(config_reader.config_path, Path)

