from manim import MathTex, Scene


class Smoke(Scene):
    def construct(self):
        self.add(MathTex(r"\text{AllGather}_X\,In[B_X, D] \to In[B, D]\quad C[I,K]\{U_X\}"))
