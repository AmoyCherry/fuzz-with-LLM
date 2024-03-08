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
        syscalls = []
        for token in top_5_tokens:
            syscalls.append(self.tokenizer.decode([token]))
        return syscalls

    def extract_syscall_name(self, syscall):
        return syscall[:syscall.index('(')]

    def testRandom(self):
        wordx = "ioctl$BTRFS_IOC_SNAP_CREATE(0xffffffffffffffff, 0x50009401, &(0x7f0000000580)={{r3}, \"b0249ef16f32e32f35759316757b5f96b16d87af3f27904d1be1d2a1c937075b591e3a40bbf953c9157885077bb58328e7b4e14dcff54fac945434326d9a282273ed8e472cbdc54ec1db098c01dcb0218713f3e2068e434290fd14e53dd02e76378a00fd09f6fa6b725705e93fcdfaa2819eb144299cb3b91a8af51a99d4cec84010f091b478df1a750f443bc36126f13a0832307dd741d1fc65fcddb87ca7b7768c5383078f2b2a56bd8eedb611c975fa2b7ec4b2757ca7b8666c3429c88e20c29557ccde26d58729ea2f900157dd78c3e5a3ec2fc4c945157ad8d3c83dbfa00cccc70b06ef11caa35ac746d8936cef558d63a4f7cbbfabbeaada273f29171ae0b5a1d4a4c82b484dc053d373aaa2568f76cf9ffd1636e37d8afc497fb1141e061acaf0fa618a62204d922d3608dc5cd961d1943904bcede37670bdfbf7b2b9aeebe845b57e7641a9a5a9d65fcab1a74bd75c7eafb1da7128b761b68d7d35ffa39af2ccb152d039d8ad36f7de607726d8a166c4aed61f0015f72ab608b6e14530b5e07309cc38bf003534e2019fc2ac2658cad951023074f3fae2f298a35f13d2d25f9899d17c1ab32fb4c140eeeec0ecab250381033695a96a29dc2a464842325c73b1a6c1524ad6cef721ddd6e72ff2382ad91d26b314aadad7ef2c91d262d63fa0f002a6f21559c12b6632267b23f875362681f08779a0f35d4b7d8a8c480675353966a24ee8bd9ec8c541d5c4d195466095a5bbc6a88f64d24be7ceb4856ef62d224b3ecaf53ed2025ed92a3caa44a2dedd900691853f04da14b4fdf5ccee717a44fdbd67090b350005b5e0de5e65ddd333da0d8765e2ecaa773c0bfb62cceac8ccb093e10fe6bccdd732ec14e76c2b4a31f3419fec1aaa4fff2ee5bbd14513e221bd029f35799bffa66b8bdec8f91102b0685978cc133567a27c2ac1359ab4a0f7971650f250fc86e3cf48c5e626c5b56527dd140c18c714f1916e65254aeca82ef97c6fa9edaf5369a925e9cae0e3a19d4d597b6cd4cce891f6ac9f283fe3c9de51fc412824ee169b5052820d6a086f1ec74377ef7b569ee3e2d9def2b04cf83db5ad1ade372ee1135d47ff86748439e8afe24a074f2ad2f28ecff68766ec123fbc7a43ef92bef5d4a3c546776b12e35ce61583df6f16a5ca9b98ca49770bade5cda1d135e0064bb2a721ed4a595115710fbc6925b4e55d3f2eccfabe1306b178fb0655d42636bf1e22de628ad89b8ac6e2e6765f9ae3064d88470a1a01361c44927da3c039add885080b41943216a71b2679ef83abe959fb9d1e458db4e256b82fdf9b2272bd0bc76c2ed20d0cf6324fa9718879a9b1b3136105476981199266b8f716ecba4ebef25109953ec6b0b15705197c84a118655b1f8b6a197fc156aa6985197ba209aeea30b859c5810abd1ae18271342306e1ae3eaf6655dfcaa109ef435e2a1e2f17ded2d099ca9dd64f7a41ec42c448600b0b5aa58931e3e836dfa9022d20d6464cb0c876ffe9bf90c52f666f8fedca6aae216675281db70c02e502d92bdda19a7dd7f117bcc2beae312fd5541b71293e52c93c3369f3bf8f8b7b135ea527c47dab9e15f39d8f04739c5633c1aee539f981a7edd664d18106bed504acdbb5319c42df77c4f14fbcdf13ac4b2403e2c513c18c83fdd73006ac5ba5d98d83fab9acef0edb7c7cab6a871531c5b6457e0f1d3547d368909ef7d5c2e3e38a816e357789e50075ecb394cb4a85a757e43197094eecd80ddada52a9d26e59a44ef620a1bed1e7fe54642a8272bf04e2522db10aac2b15f2edf37a210255313274c130a9eab0f704968d1b9430c8e6044f76f23220a94d38f6e4b7460449f6b19530749ccab430436f691585858439246f02edbed7425d679f3ecd896f172ca798b3dce005fe74e9829226262bca2879c8a6343f7e60b1cf096c0eb7ebe7dcf03353a00a51af103cbade78f9773b5d5e65f451966f3a6e0c7b047c16b358d176b141a42ecd246d334ed0e36462a5eb888c4bccd667dd77410e685a6ac5e04e1c5ae4e910c668ad704831811b1e6a6143b337e016ec1edbd63b8fb21cff11371b8b6bb4cf3b572635f36839819ba4c959ea1b523bc5112d0b90ede9b010dd034a181dde708f56f71aa20df1fac4f8e84928fae0799c5bef2d8b0846813aa19f9f04c4ebbcd11859e8fffae9ad71abc6b9ab1e73e3bd4bb53f6d18a6adfbe8c0fcf10ac5b02572b2e277bd94689d096273363748daf7e9409b5cef2b6c2fef0953a0110b9ba908fab62d20498aa350999ddbbbb9bd8f44aeda3f625dc7301a0d86a4e6d54b1e51314b9aa7425e322dd31fb47bb4ab16ac55996d9ec6aed9213d8a7e3b3a9d4738b06df53a94414619c860647bab524f3499955106dce649647e16652de649ad58f53404cd168c999a356a8e83223bd9029c176408c00d9632c8066a77b58cb5021278663f024e7333d3e500c4643577292e8a07746ef757a051a50f1bb8951b3cd61e5a4dcdfd5d5c02062ddd309ffd1254cf4aa8ba4c9f9e856b44e1a7cf8777d86bd8f68f5cb2ff57e41d13e12003a581e987f7d0ab9054364828e195fcc9bdc1a2f302e4f033a240a42e8814139dfe808cbdc96666d7f7aeb06a56d806abdfe894ff86719d2b34d15c81511bb8c7bffae8aaa8424bdfd5830e5e0c64a46e864845685bf68d8f161e5dfd9619f6f56b8a272f52ee47a729de694f29177e6047efc40383234c321346ac8c6bfa12dfa59f4facc2c3045972c06013405cb150c41e1c28772d07b072de46bc996d65a95f27ffa013850273768a69bd2e8e74ebe93f56f5dc023749f4902efbbc0d24c6230e8d7ce256b5a744a98426a4f2325ae45c7d5bebd9d19c4228def0dbf11f72937ee587b5b792deac795cea16edf708ab839c00ccc4d8b700e138a325f893ee624e13fb8a138f6de91ae250d2e31d782adb7452c0786b505aae20647b1d5a0ef7c9528fddb3f4ce36ed58c10cf9a3699518229aafc32b5ebf41d7ab4acf74629a0d749c4607ab3fcbd4d5216c0eedd021a7fae39355b4f6a68f2f2cd2f022f023c6b65d0e09f2754f130f67c2473f69a74e01e4c3ba8b51099d14a098030f67a0e24bb32eef533b3c67e7be9807a31bda5ccd70f54d6755f20f9c1ccdfeebf5ff9f5d7e853e5d22bef8832d92e58bb7daaac2a7e371b73046963fe471daa5204083adf5bc8bdb59877f0f69355a555c769104c811249dcf6fa50dffd2f64267f479ad8337d123361b598d0bd6249f7868c782b29e41512d141707e00c52c7faa4c3e7a5c0da5b0a80074caf58b1ae9600dbb453fed986c240cf1ba7b4075a46a6b5e2e8c1b8d00ca2ffbe3b1d103897e415249f5dda1050d03d1a35f2abf7b5be6463cb54fc62c9e42b85b8521957ec3b06e56b2ca3b9ab97aff2e9028d1292510a2ef2b33fe6092defb810c2cebae0b1c5b9b841ff3c94b6f441987cd6e53de9340cca4b2a9283ffba076a789faeef5035e2d225cc1b870c99e839d950cf6216c20675b3412d3dd8006ea881716c723e4407b87513cd74411bf51a4789daf74102d914952b18e10eb0e668fcf03bec0f0586301808beef9f340b7a64483afafcc63ec68c157d0f8fd0b94e5c8447078510a125b03f829514621527783ccefea067689537a0b4c8b87dd4366f803db139fbb9c3a7f8234f9bae0209bd7d77d1a58e5896aab380cb6cc5f6bf9ae7613ea32410cdc29255133480f0992d3c83cc15c99cd957f0b8bdca2d273123e9b4a3385a642f8f2cb363ab1dcfabb1e607ba3a4e01707ebc10a3c579ffc4cc20053c6dc65def6b1f8f40b84d53342e1dddb1a90455966c36ce73820eb64e921f407b550ccebb8e768952c4d665513aec0249fcc4d229dad00be65ce2ba5547c721f00a11a699cb6738d3cd072d4098a3619038f5a116c267891d885183b86286f2824542e79dcf217c227e23507e6b0c57dccf5bb9b5206c43fa5686f2a48208631456ca363414f46a6f1f2cec864d477a46a6e38e27630e62408d988fa80c04f3a9a8620f4c07efd1fb5ef762461014a17e34e6f0754f8351aaceaa18386313daee63a9576aba4783ef707ac4c8b6dc4e55b851a7489fe5c8716cc18cc7becdb4eb93987c3a1aa91dccd4d10d3483eae358661f002509ea37336bf8cd285c5f7c0895eea1ff49057e5172e121b4cb2667acf150ae1ea83b2b31407689704466b4c1c6cda3b3707d879eb09d43a9fa8ad4fedfb8d822b7f03f6d7d4a8fb86a91d93d4fe22ffecc947ba2964ecbbde2595973fb380359d8b17165c28518ef0f53831355f1aee316fefc3f11c9eaff8e47c3dcb6510d32c761a47cdd858ec81d1afca811b12ede813bf450943ebfccf5516b79d75c3fea75413243708dd46df8b9bab98bad9c033a3082a45c83b2d732e6fb34ac8277a47720d0dfab45605e364c01b084ba17097e0d72f0deda257c819d0dd6b4a170e480375612bf2ac8204cfde4d4665a765888217addd382e905038ccb5453fc2eb4dbcf71d549a9077db5c19b1d531fe3a81d2d3ad2e26e018ba5d4312197ee854bb3cad0ea1668a0bb9b2d650ad1021d71b1fd4e5fb23110b7adc832ca8b63a7fba3293a51cbd7ce6693ae2d0d5944e8674f6b54eb8b4df5419e724b138b5153aebadd37411c951398dde896c73958276e7868f22ab316ba460dec042d17ab871a19e4717e4dc8187d1fcec219e9850d9ca850b4695343d6c8b51f63f86ce3da08af8d2250324e9307931642758a7df60a52ad15ff4a9ece6d115a49f2ae2dc45e6bc34e0af6e03b145d25694177931f46c5f4424f03fbc3cc6b8e28d00c3d703776c207c876d17fa784e5b264a1ebd584d8eaa7753eacb11668c6448ab7f481d966ea620875632af509f63674b7624835a3accad0cdacf01891563fd0034495322b6ed8097dfaa6649e7fc295caee58409336b94165db315698ea831e5c784abd567a6ebb213ef8ce8462b4c663089ec2a4c1f60e22d90c51517706932d5969f252879fbfe3587552a11a7be1022d519cb3392ea9223d49debe45958817e877cfaaa39933a03c87feadc1aecbfbbb5ff897a182221b91711207f3cca2d2c1a7c58de8be32b4e3fb41598f5db3db9ce1a69c2a30b89ffe0c7661ed893dad8ed7cfbcc312d785638ddec0946e9fe45c10da5d670360d4e0e0c95c21011f9385c6e7a86a7020d491c4e1e6673639813a1cf2736a129c06369d74f9ef5e8ac1e0d54bc50a2e18031319a9cf227fcf3854f8bc0ae88fbf8c283a5ddc052ae2f72e07bf55748c76152d0a26659207c38b1a4e133f9d5eb7e4a2772802ac5677363bd3d978e949796c2dfc7d547cbfea78d2d1d2dfd158423d0a43fb5a50b708c291bc7a91159740b72a56c113ed945ab31101861f8d59812de4e31b26d23c13cf8779056684244511c2f9e50c2450668a355929cdf9024b64b0f334c7e08c3a2ea2b60222d2f24ef9eaf804c1340125e6ba28f730e1d44a7fa92499b4a5a2e8052e4a9c84d46b4d4db296ea5ef7b4c396d5fc4f37aeedc5ee3fa99acac5c2864b6c772abecee55028ec5dff971e3beaa4a0216f385cd4c463ab5a7a014a60a31c5c43cb7f95ba45ea9a3d308b3c65e907b7eb27f8db0a79b6d2812a0419142efcfcb2638059f385a0c73d98824547a2751775cdb398c38f72d9fa76b58d631301fb52c4d54765d588634afb0db410f7630cbfabc4205f88251d446444980777dffaac37bd42dacda568ff4df290c46e6b0bc65f265ce664eca743e717\"})"
        wordy= "r3 = socket$netlink(0x10, 0x3, 0x0)"
        print(self.extract_syscall_name(wordx), "\n", self.extract_syscall_name(wordy))
        sequence = [CLS, wordx, "[MASK]", wordy, SEP]
        self.GenerateNextTokenFromMask(sequence)

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

    def testTokenizeWord(self):
        word1 = "write(r0, &(0x7f0000000020)=\"010000000000000000\", 0x9)"
        word2 = "xxx"

        print(self.tokenizer.tokenize_word(word1))
        print(self.tokenizer.tokenize_word(word2))

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
