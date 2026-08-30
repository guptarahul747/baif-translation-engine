import unittest

from run_step1_translation import collapse_repeated_phrases


class TranslationCleanupTests(unittest.TestCase):
    def test_collapses_repeated_multiword_decoder_loop(self):
        text = "तीसरी कीटाणुनाशक, तीसरी कीटाणुनाशक, तीसरी कीटाणुनाशक"
        self.assertEqual(collapse_repeated_phrases(text), "तीसरी कीटाणुनाशक,")

    def test_preserves_nonrepeated_words(self):
        text = "महिला समूह के लिए बकरी पालन प्रशिक्षण आयोजित किया गया है।"
        self.assertEqual(collapse_repeated_phrases(text), text)


if __name__ == "__main__":
    unittest.main()
