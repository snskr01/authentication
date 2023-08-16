from kivy.app import App
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
import kivy.properties as kyprops
from kivy.lang.builder import Builder
import kivy
kivy.require('2.1.0')

Builder.load_file('screen.kv')

class MyScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__()
        print(kwargs)
    
    def press(self):
        input=self.root.get_screen('submit').ids.input_label.text
        self.root.ids.prompt_label.text = input


class AwesomeApp(App):
    def build(self):
        return MyScreen()

if __name__=='__main__':
    AwesomeApp().run()