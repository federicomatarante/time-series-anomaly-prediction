import math
from typing import Optional, Dict, Any


class ConfigReader:
    """
    A configuration reader class that provides convenient access to configuration data organized in sections.

    The configuration dictionary must be organized in sections, where each section contains related parameters.
    Each section is a dictionary containing key-value pairs of configuration parameters.

    Config dictionary structure:
        {
            'section1': {
                'param1': value1,
                'param2': value2
            },
            'section2': {
                'param1': value1,
                'param2': value2
            }
        }

    Examples:
         # Basic configuration setup
         config_data = {
             'app_settings': {
                 'debug_mode': True,
                 'log_level': 'INFO'
             },
             'display': {
                 'width': 800,
                 'height': 600,
                 'fullscreen': False
             }
         }
         config = ConfigReader(config_data)

         # Access configuration in different ways
         config.get_param('app_settings.debug_mode')  # Returns: True
         config.get_param('display.width')            # Returns: 800
         config['display']                            # Returns: {'width': 800, 'height': 600, 'fullscreen': False}
    """

    def __init__(self, config_data: Dict):
        """
        Initialize the ConfigReader with configuration data\
        :param config_data: Dictionary containing the configuration data organized in sections
        """
        self.config_data = config_data

    def get_param(self, param_path: str, default: Any = None, v_type: type = None) -> Any:
        """
        Get a parameter value using dot notation path with optional type casting.

        :param param_path: Path to the parameter using dot notation (e.g., 'database.host' where 'database' is the section)
        :param default: Default value to return if parameter is not found
        :param v_type: Type to cast the parameter value to (float, int, str, or bool)
        :return: The parameter value if found, otherwise the default value
        :raises ValueError: If the parameter is not found and no default provided, or if v_type is not allowed
        :raises TypeError: If type conversion fails and no default provided

        Example:
             config = ConfigReader({
                 'server': {
                     'port': '8080',
                     'host': 'localhost',
                     'debug': 'true'
                 }
             })

             config.get_param('server.port', v_type=int)  # Returns: 8080
             config.get_param('server.host')              # Returns: 'localhost'
             config.get_param('server.debug', v_type=bool)  # Returns: True
             config.get_param('server.timeout', default=30)  # Returns: 30
        """
        allowed_types = (float, int, str, bool)
        if v_type and v_type not in allowed_types:
            raise ValueError(f"v_type must be between the following categories: {allowed_types}")
        try:
            section, param = param_path.split('.')
            data = self.config_data[section][param]

            if v_type is not None:
                try:
                    # Special handling for bool type since bool('False') == True
                    if v_type is bool and isinstance(data, str):
                        data = data.lower() == 'true'
                    else:
                        data = v_type(data)
                except (ValueError, TypeError):
                    if default is None:
                        raise TypeError(
                            f"Type conversion failed for param {param_path}. Got type {type(data)}, expected: {v_type}")
                    else:
                        return default

            return data
        except (KeyError, ValueError) as e:
            if default is None:
                raise ValueError(f"Parameter {param_path} not found and no default value provided") from e
            return default

    def get_collection(self, param_path: str, default: Any = None, v_type: type = None, collection_type: type = tuple):
        """
        Get a collection parameter and optionally convert its elements to a specified type.

        :param param_path: Path to the parameter using dot notation
        :param default: Default value to return if parameter is not found
        :param v_type: Type to cast each element in the collection to. Possible values are (float, int, str, or bool).
        :param collection_type: Type of collection to return (list, tuple, or set)
        :return: The collection of values with the specified type
        :raises ValueError: If the parameter is not found and no default provided, or if collection_type is not allowed

        Example:
             config = ConfigReader({
                 'app': {
                     'ports': '[80, 443, 8080]',
                     'allowed_ips': '(192.168.1.1, 192.168.1.2)',
                     'flags': '{true, false, true}'
                 }
             })
             config.get_collection('app.ports', v_type=int)  
            # Returns: (80, 443, 8080)
             config.get_collection('app.allowed_ips', collection_type=list)  
            # Returns: ['192.168.1.1', '192.168.1.2']
             config.get_collection('app.flags', v_type=bool, collection_type=set)  
            # Returns: {True, False}
        """
        allowed_collection_types = (list, tuple, set)
        if collection_type and collection_type not in allowed_collection_types:
            raise ValueError(f"collection_type must be between the following categories: {allowed_collection_types}")
        try:
            section, param = param_path.split('.')
            data = self.config_data[section][param]
            if ((data.startswith('[') and data.endswith(']')) or
                    (data.startswith('(') and data.endswith(')')) or
                    (data.startswith('{') and data.endswith('}'))):
                data = data.strip('{[()]}')
                data = data.split(',')
                data = [d.rstrip(' ').lstrip(' ') for d in data]
                if not v_type:
                    return collection_type(data)
                allowed_v_types = (float, int, str, bool)
                if v_type and v_type not in allowed_v_types:
                    raise ValueError(
                        f"v_type must be between the following categories: {allowed_v_types}")
                return collection_type(v_type(sample) for sample in data) if v_type != bool else collection_type(
                    sample.lower() == 'true' for sample in data)
        except (KeyError, ValueError) as e:
            if default is None:
                raise ValueError(f"Parameter {param_path} not found and no default value provided") from e
            return default

    def get_section(self, section: str) -> Optional[Dict[str, str]]:
        """
        Get all parameters in a section.

        :param section: Name of the configuration section
        :return: Dictionary containing all parameters in the section, or None if not found

        Example:
             config = ConfigReader({
                 'database': {
                     'host': 'localhost',
                     'port': 5432,
                     'username': 'admin'
                 }
             })
             config.get_section('database')  
            # Returns: {'host': 'localhost', 'port': 5432, 'username': 'admin'}
             config.get_section('invalid')   
            # Returns: None
        """
        return self.config_data.get(section)

    def __getitem__(self, key: str) -> Dict[str, str]:
        """
        Allow dictionary-style access to configuration sections\
        :param key: Section name
        :return: Dictionary containing all parameters in the section
        :raises KeyError: If the section is not found
        """
        return self.config_data[key]
