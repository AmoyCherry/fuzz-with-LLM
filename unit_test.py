from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, top_k_top_p_filtering
import torch

import unittest

from syz_tokenizer import SyzTokenizer
from utils import ModelPath


word1 = 'r0 = socket$inet_udplite(0x2, 0x2, 0x88)'
word2 = 'ioctl(r0, 0x80001000008912, &(0x7f0000000000)=\"0adc1f123c123f319bd070\")'
word3 = 'keyctl$instantiate_iov(0x14, 0x0, &(0x7f00000001c0)=[{&(0x7f0000000380)=\"d91a86229048808fbb8e2cddebc99a1db0566d42619f1ff19963613f0231b96e5c6460d25b0351770bcd84d52052a24db4188983e6c32c4f89ed532c91717ad7344bf89d03dcdf35e3844083fcb2d95ac55f15a5e5e28dd710d3fb3be8ea1770a6f4c4c9ef1a45de1154\", 0x6a}], 0x1, 0x0)'
word4 = 'sendmsg$key(0xffffffffffffffff, &(0x7f00000000c0)={0x0, 0x0, &(0x7f0000000080)={&(0x7f0000000480)=ANY=[@ANYBLOB=\"02000000b50000000066dc000000005188adfeb4cc55070002d9f133c0616c072a20139b9ac715e5c07c0227c829273502eb42cd8030f39022e15b9479d77bfc098beef32dc3fabf1cc4ad639e0db270328b9270a90532b371c01c50d9abf9ded461f8319a48bd42008752f1e5b2ccbbc331f3ca01fb8628e59c96f717025bfdf19d6e84829c0f852e5e3cc1c45e7e052f5d00005bfeeb22320a261ff07f86bc9832852060aa92d7b2005800b4fbd8cfbeaf42e14ae4e8a990bb4ce7f466aabd0400e133a9462f854753bb2ff779e1de785d7a341549d967739166cacab9d466703695133fdc5e9e1cb0413f99ad32fc90166a40\"], 0xf4}}, 0x0)'
word5 = 'syz_execute_func(&(0x7f0000000240)=\"f2af91cd800f0124eda133fa20430fbafce842f66188d0d038c4ab39fd5bf9e2f9e2c7c7e4c653fb0fc4014cb63a3af4a95bf9c44149f2168f4808eebce00000802000c863fa43adc4e17a6fe60f186746f340aee47c7c730f66400f3833fe8f0f14e7e701fe5ff6e7df660fe7af5cc34a510804f4c441a5609c8ba80000005499\")'


worda = "r0 = openat(0xffffffffffffff9c, &(0x7f0000000000)='/proc/self/exe\\x00', 0x0, 0x0)"
wordb = 'mmap(&(0x7f0000000000/0x800000)=nil, 0x800000, 0x1800003, 0x12, r0, 0x0)'

MASK = "[MASK]"
CLS = "[CLS]"
SEP = "[SEP]"


class TestSyzTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SyzTokenizer()
        self.mask_model = AutoModelForMaskedLM.from_pretrained(ModelPath)
        #self.causal_model = AutoModelForCausalLM.from_pretrained(ModelPath)
        #self.causal_model = TFAutoModelForCausalLM.from_pretrained(ModelPath)

    def tearDown(self):
        self.tokenizer = None
        self.model = None

    def GenerateNextTokenFromMask(self, sequence):
        input_ids_tensor = self.tokenizer.tokenize_sequence(sequence, return_tensors="pt")
        input_ids = input_ids_tensor.data['input_ids']
        mask_token_index = torch.where(input_ids == 211693)[1]
        mask_token_logits = self.mask_model(input_ids).logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        resulting_string = self.tokenizer.decode(top_5_tokens)
        print("top 5 tokens for next:\n\n", resulting_string)

    def testGenerateNextTokenFromMask(self):
        print("mask socket: ")
        sequence1 = [CLS, MASK, word2, word3, word4, word5, SEP]
        self.GenerateNextTokenFromMask(sequence1)

        print("mask ioctl: ")
        sequence2 = [CLS, word1, MASK, word3, word4, word5, SEP]
        self.GenerateNextTokenFromMask(sequence2)

        print("mask keyctl: ")
        sequence3 = [CLS, word1, word2, MASK, word4, word5, SEP]
        self.GenerateNextTokenFromMask(sequence3)

        print("mask sendmsg: ")
        sequence4 = [CLS, word1, word2, word3, MASK, word5, SEP]
        self.GenerateNextTokenFromMask(sequence4)

        print("mask sendmsg: ")
        sequence4 = [CLS, word1, word2, MASK, word5, SEP]
        self.GenerateNextTokenFromMask(sequence4)

        print("mask sendmsg: ")
        sequence4 = [CLS, word1, word2, MASK, SEP]
        self.GenerateNextTokenFromMask(sequence4)

        print("mask syz_execute_func: ")
        sequence5 = [CLS, word1, word2, word3, word4, MASK, SEP]
        self.GenerateNextTokenFromMask(sequence5)

        sequence6 = [CLS, worda, wordb, MASK, SEP]
        self.GenerateNextTokenFromMask(sequence6)


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
