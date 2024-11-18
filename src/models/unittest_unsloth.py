import unittest
from src.models.unsloth_model import UnslothModel, UnslothConfig

class TestUnsloth(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.config = UnslothConfig()
        self.model = UnslothModel(model_config=self.config)

    def test_generate(self):
        try:
            self.model.generate("Test prompt")
        except Exception as e:
            self.fail(f"generate method threw an exception: {e}")
        

    def test_chat(self):
        try:
            session = self.model.get_chat_session()
        except Exception as e:
            self.fail(f"get_chat_session method threw an exception: {e}")

        try:
            response1 = session.chat("don't repeat yourself, say another thing if I repeat myself")
        except Exception as e:
            self.fail(f"ChatSession.chat method threw an exception: {e}")

        try:
            response2 = session.chat("don't repeat yourself, say another thing if I repeat myself")
        except Exception as e:
            self.fail(f"second ChatSession.chat method call threw an exception: {e}")

        self.assertNotEqual(response1, response2)

if __name__ == '__main__':
    unittest.main()