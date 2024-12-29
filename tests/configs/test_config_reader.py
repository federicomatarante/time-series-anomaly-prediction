import unittest
from typing import Dict, Any

from src.utils.config.config_reader import ConfigReader


class TestConfigReader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.test_config: Dict[str, Dict[str, Any]] = {
            'app_settings': {
                'debug_mode': 'true',
                'log_level': 'INFO',
                'max_connections': '100',
                'timeout': '30.5'
            },
            'display': {
                'width': '800',
                'height': '600',
                'fullscreen': 'false',
                'refresh_rate': '60'
            },
            'server': {
                'ports': '[80, 443, 8080]',
                'allowed_ips': '(192.168.1.1, 192.168.1.2)',
                'enabled_flags': '{true, false, true}'
            }
        }
        self.config_reader = ConfigReader(self.test_config)

    def test_get_param_basic(self):
        """Test basic parameter retrieval without type conversion"""
        self.assertEqual(self.config_reader.get_param('app_settings.log_level'), 'INFO')
        self.assertEqual(self.config_reader.get_param('display.width'), '800')

    def test_get_param_with_type_conversion(self):
        """Test parameter retrieval with type conversion"""
        self.assertEqual(self.config_reader.get_param('app_settings.max_connections', v_type=int), 100)
        self.assertEqual(self.config_reader.get_param('app_settings.timeout', v_type=float), 30.5)
        self.assertTrue(self.config_reader.get_param('app_settings.debug_mode', v_type=bool))
        self.assertFalse(self.config_reader.get_param('display.fullscreen', v_type=bool))

    def test_get_param_with_default(self):
        """Test parameter retrieval with default values"""
        self.assertEqual(
            self.config_reader.get_param('app_settings.invalid', default='DEFAULT'),
            'DEFAULT'
        )
        self.assertEqual(
            self.config_reader.get_param('invalid.param', default=123),
            123
        )

    def test_get_param_invalid_type(self):
        """Test parameter retrieval with invalid type conversion"""
        with self.assertRaises(ValueError):
            self.config_reader.get_param('app_settings.log_level', v_type=dict)

    def test_get_param_type_conversion_error(self):
        """Test parameter retrieval with failed type conversion"""
        with self.assertRaises(TypeError):
            self.config_reader.get_param('app_settings.log_level', v_type=int)

    def test_get_collection_basic(self):
        """Test basic collection retrieval"""
        expected_ports = (80, 443, 8080)
        self.assertEqual(
            self.config_reader.get_collection('server.ports', v_type=int),
            expected_ports
        )

    def test_get_collection_with_different_types(self):
        """Test collection retrieval with different collection types"""
        # Test list type
        ports_list = self.config_reader.get_collection(
            'server.ports',
            v_type=int,
            collection_type=list
        )
        self.assertIsInstance(ports_list, list)
        self.assertEqual(ports_list, [80, 443, 8080])

        # Test set type
        flags_set = self.config_reader.get_collection(
            'server.enabled_flags',
            v_type=bool,
            collection_type=set
        )
        self.assertIsInstance(flags_set, set)
        self.assertEqual(flags_set, {True, False})

    def test_get_collection_with_default(self):
        """Test collection retrieval with default values"""
        default_value = [1, 2, 3]
        result = self.config_reader.get_collection(
            'server.invalid',
            default=default_value,
            collection_type=list
        )
        self.assertEqual(result, default_value)

    def test_get_collection_invalid_type(self):
        """Test collection retrieval with invalid collection type"""
        with self.assertRaises(ValueError):
            self.config_reader.get_collection('server.ports', collection_type=dict)

    def test_get_section(self):
        """Test retrieving entire configuration sections"""
        display_section = self.config_reader.get_section('display')
        self.assertEqual(display_section, self.test_config['display'])

        # Test non-existent section
        self.assertIsNone(self.config_reader.get_section('invalid_section'))

    def test_dictionary_access(self):
        """Test dictionary-style access to configuration sections"""
        self.assertEqual(
            self.config_reader['app_settings'],
            self.test_config['app_settings']
        )

        with self.assertRaises(KeyError):
            _ = self.config_reader['invalid_section']

    def test_param_not_found(self):
        """Test behavior when parameter is not found"""
        with self.assertRaises(ValueError):
            self.config_reader.get_param('invalid.param')

    def test_invalid_param_path(self):
        """Test behavior with invalid parameter paths"""
        invalid_paths = [
            'invalid_path',
            'too.many.dots.here',
            '.nodots',
            'dots.',
            ''
        ]

        for path in invalid_paths:
            with self.assertRaises(ValueError):
                self.config_reader.get_param(path)

