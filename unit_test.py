from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, top_k_top_p_filtering
from torch.nn import functional as F
import torch

import unittest

from syz_tokenizer import SyzTokenizer
from utils import ModelPath


word1 = 'r0 = socket$inet_udplite(0x2, 0x2, 0x88)'
word2 = 'ioctl(r0, 0x80001000008912, &(0x7f0000000000)=\"0adc1f123c123f319bd070\")'

word3 = 'read(r0, &(0x7f000000001e)=\"\"/33, 0x21)'
word4 = 'pipe(&(0x7f0000000008)={<r0=>0xffffffffffffffff})'
word5 = "open(&(0x7f0000000000)='/lib/x86_64-linux-gnu/libc.so.6\\x00', 0x80000, 0x0)"
word6 = "open(&(0x7f0000000000)='/etc/ld.so.cache\\x00', 0x80000, 0x0)"
MASK = "[MASK]"
CLS = "[CLS]"
SEP = "[SEP]"

word7 = 'r0 = socket$inet_udplite(0x2, 0x2, 0x88)'
word8 = "sendmsg$TIPC_NL_NODE_GET(0xffffffffffffffff, &(0x7f0000000240)={0x0, 0x0, &(0x7f0000000200)={&(0x7f0000000000)={0x3c, 0x0, 0x0, 0x0, 0x0, {}, [@TIPC_NLA_LINK={0x28, 0x4, [@TIPC_NLA_LINK_NAME={0xc, 0x1, 'syz0\\x00'}, @TIPC_NLA_LINK_NAME={0xc, 0x1, 'syz1\\x00'}, @TIPC_NLA_LINK_NAME={0xc, 0x1, 'syz1\\x00'}]}]}, 0x3c}}, 0x0)"



class TestSyzTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SyzTokenizer()
        self.mask_model = AutoModelForMaskedLM.from_pretrained(ModelPath)
        self.causal_model = AutoModelForCausalLM.from_pretrained(ModelPath)
        # self.causal_model = TFAutoModelForCausalLM.from_pretrained(ModelPath)

    def tearDown(self):
        self.tokenizer = None
        self.model = None

    def testGenerateNextTokenFromMask(self):
        sequence1 = [CLS, word1, MASK, word8, SEP]

        input_ids_tensor = self.tokenizer.tokenize_sequence(sequence1, return_tensors="pt")
        input_ids = input_ids_tensor.data['input_ids']
        mask_token_index = torch.where(input_ids == 211693)[1]
        mask_token_logits = self.mask_model(input_ids).logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        resulting_string = self.tokenizer.decode(top_5_tokens)
        print("\n top 5 tokens for next:\n\n", resulting_string)

    #def textGenerateNextTokenFromCausal(self):

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
    suite.addTest(TestSyzTokenizer("testGenerateNextTokenFromMask"))

    return suite


if __name__ == "__main__":
    unittest.main(defaultTest=suite())
