"""Parse and provide data defined in properties file."""

from jproperties import Properties

class Config(Properties):
    """Application configuration properties class."""

    # CONSTRUCTOR =================================================================================

    def __init__(self):
        """Initialize Config object."""
        super().__init__()

        with open('trader.properties', 'rb') as properties:
            self.load(properties)

    # VERIFIERS ===================================================================================

    def has_property(self, property: str) -> bool:
        """Indicate existence of specified property.

        Args:
            property (str): Property search key

        Returns:
            bool: True if property exists, False otherwise
        """
        return property in self._properties

    # GETTERS =====================================================================================

    def get_api_key(self) -> str:
        """Provide API key.

        Returns:
            str: API key
        """
        return self.get('api_key').data
    
    # DUNDERS =====================================================================================
    def __call__(self):
        """Call __str__.
        """
        return self.__str__()
    
    def __str__(self) -> str:
        """Provide str format of Config object.

        Returns:
            str: String format
        """
        # Intialize str with header
        config_str = 'Properties:'

        # Build list
        for key in self._properties:
            config_str += f"\n{key:<15}{self._properties[key]:>15}"

        # Return str format
        return config_str

# TESTING =========================================================================================

if __name__ == '__main__':
    """Run tests for Config class.
    """
    print("============\n"
          "CONFIG TESTS\n"
          "============\n")

    # Intialize Config class
    config = Config()

    # Print config to see properties recorded
    print(config)

    # Check existence of property 'api_key'
    print(f"\nHas \'api_key\': {config.has_property('api_key')}")