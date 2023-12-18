from Trainer import SyzTokenizer

import unittest


class TestSyzTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SyzTokenizer()

    def tearDown(self):
        self.tokenizer = None

    #def testTokenFileShouldMappingToTokenizer(self):


    def testTokenizeSequence(self):
        word1 = "mq_unlink(&(0x7f0000000000)='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\x00')"
        word2 = "write(r0, &(0x7f0000000020)=\"010000000000000000\", 0x9)"
        word3 = "open(&(0x7f0000000018)='dupfile\\x00', 0x42, 0x1c0)"

        self.assertTokenizeSequence([word1])
        self.assertTokenizeSequence([word1, word2])
        self.assertTokenizeSequence([word1, word2, word3])

    def assertTokenizeSequence(self, sequence: list[str]):
        tokens = self.tokenizer.tokenize_sequence(sequence)

        #   tokenizer will add [CLS] - 3301 and [SEP] - 3299 to the two ends
        for i in range(len(sequence)):
            word = sequence[i]
            id = tokens.input_ids[i + 1]
            self.assertEqual(id, self.tokenizer.tokenize_word(word))


def suite():
    suite = unittest.TestSuite()

    suite.addTest(TestSyzTokenizer("testTokenizeSentence"))

    return suite


if __name__ == "__main__":
    unittest.main(defaultTest=suite())
