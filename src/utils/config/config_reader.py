from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set


class ConfigReader:
    """
    A configuration reader for accessing structured configuration data with type safety and validation.

        :param config_data: A nested dictionary containing configuration data organized in sections.
                          The top-level keys represent sections, and their values are dictionaries
                          containing the configuration parameters.

    The ConfigReader class provides methods to access configuration parameters organized in sections,
    with support for type conversion, default values, and collection handling. The configuration data
    should be provided as a nested dictionary where the top-level keys represent sections and their
    values are dictionaries containing the actual configuration parameters.

    Configuration Structure:
        The config_data dictionary should follow this structure:
            {
                'section1': {
                    'param1': 'value1',
                    'param2': 'value2'
                },
                'section2': {
                    'param3': 'value3',
                    'param4': 'value4'
                }
            }

    Key Features:
        - Dot notation access to nested parameters
        - Type conversion with validation
        - Collection handling (lists, tuples, sets)
        - Default value support
        - Null value handling
        - Section-level access

    Examples:
         # Initialize with configuration data
         config_data = {
             'database': {
                 'host': 'localhost',
                 'port': '5432',
                 'max_connections': '100'
             },
             'api': {
                 'timeout': '30',
                 'retry_count': '3',
                 'endpoints': '[/v1, /v2, /v3]'
             }
         }
         config = ConfigReader(config_data)

         # Get parameters with type conversion
         host = config.get_param('database.host')
         port = config.get_param('database.port', v_type=int)
         timeout = config.get_param('api.timeout', v_type=int, default=60)

         # Get collections
         endpoints = config.get_collection('api.endpoints', collection_type=list)

         # Access entire sections
         db_config = config.get_section('database')
         # or using dictionary syntax
         db_config = config['database']
    """

    def __init__(self, config_data: Dict, base_path: Path = Path('')):
        """
        Initialize the ConfigReader with configuration data\
        :param config_data: Dictionary containing the configuration data organized in sections
        """
        self.config_data = config_data
        self.base_path = base_path

    def _convert_type(self, data: str, v_type: type, param_path: str, default=None, domain: set = None):
        allowed_types = (float, int, str, bool, Path)

        if v_type and v_type not in allowed_types:
            raise ValueError(f"v_type must be between the following categories: {allowed_types}")
        try:
            # Special handling for bool type since bool('False') == True
            if v_type is bool:
                value = data.lower() == 'true'
            elif v_type is Path:
                value = self.base_path / Path(data)
                print(value)
            else:
                value = v_type(data)

        except (ValueError, TypeError):
            if default is None:
                raise TypeError(
                    f"Type conversion failed for param {param_path}. Got type {type(data)}, expected: {v_type}")
            else:
                return default
        if domain and value not in domain:
            raise ValueError(f"Parameter {param_path} must be one of the following values: {domain}. Found: {value}")
        return value

    def get_param(self, param_path: str, v_type: type = None, default: Any = None, nullable: bool = False,
                  domain: set = None) -> Any:
        """
        Retrieve a configuration parameter using dot notation with optional type conversion.

        :param param_path: Path to the parameter using dot notation (e.g., 'database.host')
        :param default: Value to return if parameter is not found
        :param v_type: Type to convert the parameter value to (float, int, str, or bool)
        :param nullable: Whether to allow null values ('null' or 'none')
        :param domain: set of allowed values.
        :return: The parameter value converted to the specified type if applicable
        :raises ValueError: If the parameter is not found and no default is provided,
                          or if v_type is not supported or if the value is not in the domain.
        :raises TypeError: If type conversion fails and no default is provided

        Example:
            config_data = {
                'server': {
                    'host': 'localhost',
                    'port': '8080',
                    'debug': 'true',
                    'timeout': 'null'
                }
            }
            config = ConfigReader(config_data)

            # Basic parameter access
            host = config.get_param('server.host')  # Returns: 'localhost'

            # Type conversion
            port = config.get_param('server.port', v_type=int)  # Returns: 8080
            debug = config.get_param('server.debug', v_type=bool)  # Returns: True

            # Default values
            rate = config.get_param('server.rate', default=60.0, v_type=float)  # Returns: 60.0

            # Null values
            timeout = config.get_param('server.timeout', nullable=True)  # Returns: None
        """
        if '.' not in param_path:
            raise ValueError(f"Parameter {param_path} must be in the format [section].[key]")
        try:
            section, param = param_path.rsplit('.', 1)
            data = str(self.config_data[section][param])
            if data.lower() in ('null', 'none'):
                if nullable:
                    return None
                raise ValueError(f"Parameter {param_path} cannot be a null value!")
            if v_type is not None:
                data = self._convert_type(data, v_type, param_path, default, domain)
            return data
        except KeyError as e:
            if default is None:
                if nullable:
                    return None
                raise ValueError(f"Parameter {param_path} not found and no default value provided") from e
            return default

    def get_collection(self, param_path: str, default: Any = None, v_type: type = None, collection_type: type = tuple,
                       nullable: bool = False, num_elems: int = None, domain: set = None):
        """
        Retrieve and parse a collection parameter with optional type conversion for its elements.

        :param param_path: Path to the parameter using dot notation
        :param default: Value to return if parameter is not found
        :param v_type: Type to convert each collection element to (float, int, str, or bool)
        :param collection_type: Type of collection to return (list, tuple, or set)
        :param nullable: Whether to allow null values ('null' or 'none')
        :param num_elems: Expected number of elements in the collection
        :param domain: set of allowed values.
        :return: The parsed collection of the specified type
        :raises ValueError: If the parameter is not found and no default provided,
                          if collection_type is not supported,
                          or if num_elems doesn't match actual length

        Example:
            config_data = {
                'app': {
                    'ports': '[80, 443, 8080]',
                    'hosts': '(host1, host2)',
                    'flags': '{true, false, true}',
                    'coords': '[10, 20]',
                    'empty': 'null'
                }
            }
            config = ConfigReader(config_data)

            # Basic collection access
            ports = config.get_collection('app.ports', v_type=int)  # Returns: (80, 443, 8080)

            # Different collection types
            hosts = config.get_collection('app.hosts', collection_type=list)  # Returns: ['host1', 'host2']

            # Boolean values in a set
            flags = config.get_collection('app.flags', v_type=bool, collection_type=set)  # Returns: {True, False}

            # Validate number of elements
            coords = config.get_collection('app.coords', v_type=int, num_elems=2)  # Returns: (10, 20)
        """
        if '.' not in param_path:
            raise ValueError(f"Parameter {param_path} must be in the format [section].[key]")
        allowed_collection_types = (list, tuple, set)
        if collection_type and collection_type not in allowed_collection_types:
            raise ValueError(f"collection_type must be between the following categories: {allowed_collection_types}")
        try:
            section, param = param_path.rsplit('.', 1)
            data = str(self.config_data[section][param])
            if data.lower() in ('null', 'none'):
                if nullable:
                    return None
                raise ValueError(f"Parameter {param_path} cannot be a null value!")
            if ((data.startswith('[') and data.endswith(']')) or
                    (data.startswith('(') and data.endswith(')')) or
                    (data.startswith('{') and data.endswith('}'))):
                data = data.strip('{[()]}')
                data = data.split(',')
                data = [d.rstrip(' ').lstrip(' ') for d in data]
                if num_elems and len(data) != num_elems:
                    raise ValueError(
                        f"Parameter {param_path} must be a collection of {num_elems}. {len(data)} elements found instead!!")
                if not v_type:
                    return collection_type(data)
                return collection_type(self._convert_type(sample, v_type, param_path, default) for sample in data)
            else:
                raise ValueError(
                    f"Parameter {param_path} must be a collection of elements. Use the notation [ elem_1, elem_2, ... ].")
        except KeyError as e:
            if default is None:
                raise ValueError(f"Parameter {param_path} not found and no default value provided") from e
            return default

    def get_section(self, section: str) -> Optional[Dict[str, str]]:
        """
        Retrieve all parameters in a configuration section.

        :param section: Name of the configuration section to retrieve
        :return: Dictionary containing all parameters in the section, or None if section is not found

        Example:
            config_data = {
                'database': {
                    'host': 'localhost',
                    'port': '5432',
                    'username': 'admin'
                }
            }
            config = ConfigReader(config_data)

            # Get existing section
            db_config = config.get_section('database')
            # Returns: {'host': 'localhost', 'port': '5432', 'username': 'admin'}

            # Get non-existent section
            invalid = config.get_section('invalid')  # Returns: None
        """
        return self.config_data.get(section)

    def sub_reader(self, sections: Set[str] | Tuple[str] | List[str]):
        """
        Create a new ConfigReader instance with a subset of the original configuration sections.

        :param sections: Collection of section names to include in the new ConfigReader instance.
                        Can be provided as a Set, Tuple, or List of strings.
        :return: A new ConfigReader instance containing only the specified sections

        Example:
            config_data = {
                'database': {
                    'host': 'localhost',
                    'port': '5432'
                },
                'api': {
                    'timeout': '30'
                },
                'logging': {
                    'level': 'INFO'
                }
            }
            config = ConfigReader(config_data)

            # Create a sub-reader with specific sections
            db_api_config = config.sub_reader({'database', 'api'})
            # Returns a ConfigReader with only 'database' and 'api' sections

            # Access parameters in the sub-reader
            host = db_api_config.get_param('database.host')  # Returns: 'localhost'
        """
        new_config = {
            section: self.config_data.get(section) for section in sections
        }
        return ConfigReader(new_config)

    def __getitem__(self, key: str) -> Dict[str, str]:
        """
        Enable dictionary-style access to configuration sections.

        :param key: Name of the configuration section to retrieve
        :return: Dictionary containing all parameters in the section
        :raises KeyError: If the section is not found

        Example:
            config_data = {
                'cache': {
                    'type': 'redis',
                    'ttl': '3600'
                }
            }
            config = ConfigReader(config_data)

            # Access section using dictionary syntax
            cache_config = config['cache']  # Returns: {'type': 'redis', 'ttl': '3600'}

            # KeyError for missing section
            try:
                invalid = config['invalid']
            except KeyError as e:
                print(str(e))  # "'invalid'"
        """
        return self.config_data[key]
