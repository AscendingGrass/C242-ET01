import unittest
from src.models.gemini import Gemini, GeminiConfig

class TestGemini(unittest.TestCase):
    def setUp(self):
        self.config = GeminiConfig(temperature=0)
        self.gemini = Gemini(model_config=self.config)

    def test_generate(self):
        try:
            self.gemini.generate("Test prompt", return_response=False)
        except Exception as e:
            self.fail(f"generate method threw an exception: {e}")
        

    def test_chat(self):
        try:
            session = self.gemini.get_chat_session()
        except Exception as e:
            self.fail(f"get_chat_session method threw an exception: {e}")

        try:
            response1 = session.chat("Test prompt", return_response=False)
        except Exception as e:
            self.fail(f"ChatSession.chat method threw an exception: {e}")

        try:
            response2 = session.chat("Test prompt", return_response=False)
        except Exception as e:
            self.fail(f"second ChatSession.chat method call threw an exception: {e}")

        self.assertNotEqual(response1, response2)

if __name__ == '__main__':
    unittest.main()