from kivy.app import App
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
import kivy.properties as kyprops
from kivy.lang import Builder
Builder.load_file('screen.kv')

class MyLayout(Widget):
    def press(self):
        input=self.ids.input_label.text
        self.ids.prompt_label.text = input


class AwesomeApp(App):
    def build(self):
        return MyLayout()
if __name__=='__main__':
    AwesomeApp.run