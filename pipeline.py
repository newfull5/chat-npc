from langgraph.graph import StateGraph, END

class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def sentinel(self):
        pass

