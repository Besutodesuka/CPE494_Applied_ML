#!/usr/bin/env python
import kivy
kivy.require('2.2.0')

from kivy.app import App
from kivy.uix.label import Label

class TestApp(App):
    def build(self):
        return Label(text='GUI Test Working!')

if __name__ == '__main__':
    TestApp().run()