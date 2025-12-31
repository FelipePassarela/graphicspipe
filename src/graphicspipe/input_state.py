from pynput import keyboard


class InputState:
    def __init__(self):
        self._keys = set()

    def on_press(self, key: keyboard.KeyCode | keyboard.Key):
        try:
            self._keys.add(key.char.lower())
        except AttributeError:
            self._keys.add(key)

    def on_release(self, key: keyboard.KeyCode | keyboard.Key):
        try:
            self._keys.discard(key.char.lower())
        except AttributeError:
            self._keys.discard(key)

    def is_pressed(self, key: keyboard.KeyCode | keyboard.Key) -> bool:
        try:
            return key.char.lower() in self._keys
        except AttributeError:
            return key in self._keys
