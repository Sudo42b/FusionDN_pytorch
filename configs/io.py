import toml
import re
import os


class TOMLConfig:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = self.load_config()
        self.resolve_references()

    def load_config(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        with open(filepath, 'r', encoding='utf8') as file:
            return toml.load(file)

    def resolve_references(self):
        # Recursively resolve references within the entire config
        self.config = self.resolve_value(self.config)

    def resolve_value(self, value, current_file=None):
        if isinstance(value, dict):
            # Recursively resolve all values in a dictionary
            return {k: self.resolve_value(v, current_file) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively resolve values in a list
            return [self.resolve_value(v, current_file) for v in value]
        elif isinstance(value, str):
            # Resolve references within a string value
            pattern = r"\$\{([^}]+)\}"
            while True:
                match = re.search(pattern, value)
                if not match:
                    break
                ref_path = match.group(1)
                ref_value = self.get_ref_value(ref_path, current_file)
                if isinstance(ref_value, str):
                    value = value.replace(match.group(0), ref_value)
                else:
                    raise ValueError(f"Reference '{ref_path}' cannot be resolved to a string.")
        return value

    def get_ref_value(self, ref_path, current_file=None):
        """
        Resolve a reference in the form of 'key' or 'filename.key'.
        """
        parts = ref_path.split('.')

        # If reference contains a file path (e.g., 'otherfile.key')
        if len(parts) > 1:
            ref_file = parts[0]
            ref_key_path = parts[1:]

            # Resolve the path relative to the current file
            if current_file:
                base_dir = os.path.dirname(current_file)
                ref_file_path = os.path.join(base_dir, ref_file)
            else:
                ref_file_path = ref_file

            # Load the referenced TOML file
            ref_config = self.load_config(ref_file_path)

            # Resolve the key in the referenced file
            ref_value = ref_config
            for part in ref_key_path:
                ref_value = ref_value.get(part)
                if ref_value is None:
                    raise ValueError(f"Reference '{ref_path}' not found in '{ref_file_path}'.")

            return self.resolve_value(ref_value, ref_file_path)

        # If reference is within the same file (single key)
        else:
            ref_value = self.config
            for part in parts:
                ref_value = ref_value.get(part)
                if ref_value is None:
                    raise ValueError(f"Reference '{ref_path}' not found in configuration.")
            return self.resolve_value(ref_value, current_file)

    def __getitem__(self, item):
        return self.config[item]

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def __repr__(self):
        return '\n'.join([f'{key} = {value}' for key, value in self.config.items()])

    def save(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        with open(filepath, 'w', encoding='utf8') as file:
            toml.dump(self.config, file)
