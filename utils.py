import json
import os

PRESETS_FILE = os.path.join(os.path.dirname(__file__), "presets.json")

class PresetManager:
    def __init__(self):
        self.filepath = PRESETS_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({}, f, indent=4)

    def load_presets(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_presets(self, data):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def get_categories(self):
        data = self.load_presets()
        return list(data.keys())

    def get_keys(self, category):
        data = self.load_presets()
        return list(data.get(category, {}).keys())

    def get_value(self, category, key):
        data = self.load_presets()
        return data.get(category, {}).get(key, "")

    def add_category(self, category):
        data = self.load_presets()
        if category not in data:
            data[category] = {}
            self.save_presets(data)
            return True
        return False

    def save_preset(self, category, key, value):
        data = self.load_presets()
        if category not in data:
            data[category] = {}
        data[category][key] = value
        self.save_presets(data)
        return True

    def delete_preset(self, category, key):
        data = self.load_presets()
        if category in data and key in data[category]:
            del data[category][key]
            self.save_presets(data)
            return True
        return False

    def delete_category(self, category):
        data = self.load_presets()
        if category in data:
            del data[category]
            self.save_presets(data)
            return True
        return False
