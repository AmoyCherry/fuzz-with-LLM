import unittest

import torch

from syzLLM_server import validate_syscall, tokenizer, highest_power_of_2, mask_model, extract_syscall_name


class TestSyzllmServer(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_topk(self):
        tests = [
            {
                'name': 'topk',
                'input_calls':
"""write$SyzLLM(0x3, 0x55b312158960, 0x4c)
clock_gettime$SyzLLM(0x7)
select$SyzLLM(0xc, &(0x7f00000ea000)={0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1}, &(0x7f00000ea400)={0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1}, &(0x7f00000ea800)={0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1}, &(0x7f00000eac00)=nil)
read$SyzLLM(0x17, &(0x7f0000017000)={0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x1, ""/64}, 0x1)
futex$SyzLLM(&(0x7f0000041000)=0x1, 0x80, 0x0, &(0x7f0000041400)=nil, &(0x7f0000041800)=0x1, 0x0)
gettid$SyzLLM()
futex$SyzLLM(&(0x7f0000041000)=0x1, 0x81, 0x1, &(0x7f0000041400)=nil, &(0x7f0000041800)=0x1, 0x0)
write$SyzLLM(0x18, &(0x7f000000a000), 0x1)
futex$SyzLLM(&(0x7f0000041000)=0x1, 0x81, 0x1, &(0x7f0000041400)=nil, &(0x7f0000041800)=0x1, 0x0)
newstat$SyzLLM("/home/server/.mozilla/firefox/havhuwiy.default-release/saved-telemetry-pings/8904950c-3832-4023-8010-8474339bd248")
[MASK]
read$SyzLLM(0x6, &(0x7f0000017000)={0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x1, ""/64}, 0x1)
read$SyzLLM(0x6, &(0x7f0000017000)={0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x1, ""/64}, 0x1)
rt_sigprocmask$SyzLLM(0x0, &(0x7f0000074000), &(0x7f0000074400)=nil, 0x8)
rt_sigprocmask$SyzLLM(0x2, &(0x7f0000074000), &(0x7f0000074400)=nil, 0x8)
wait4$SyzLLM(0x7c5, 0x3)""",
            },
            {
                'name': 'munmap$SyzLLM',
                'input_calls':
"""mprotect$SyzLLM(&(0x7f000003d000/0x1000)=nil, 0x1000, 0x1)
getpid$SyzLLM()
mprotect$SyzLLM(&(0x7f000003d000/0x1000)=nil, 0x1000, 0x1)
openat$SyzLLM(0xffffffffffffff9c, &(0x7f0000008000)=nil, 0xa00c2, 0x180)
openat$SyzLLM(0xffffffffffffff9c, &(0x7f0000008000)=nil, 0xa0000, 0x0)
unlink$SyzLLM(&(0x7f0000022000)='./file0\x00')
ftruncate$SyzLLM(@RSTART@openat$SyzLLM(0xffffffffffffff9c, &(0x7f0000008000)=nil, 0xa00c2, 0x180)@REND@, 0x661a0)
mmap$SyzLLM(0x0, 0x661a0, 0x3, 0x1, @RSTART@openat$SyzLLM(0xffffffffffffff9c, &(0x7f0000008000)=nil, 0xa00c2, 0x180)@REND@, 0x0)
mprotect$SyzLLM(&(0x7f000003d000/0x1000)=nil, 0x8000, 0x1)
[MASK]""",
            }
        ]

        for test in tests:
            input_calls = test['input_calls']
            call_list = input_calls.split('\n')
            TestSyzllmServer.mock_server(call_list)

    @staticmethod
    def mock_server(syscall_list):
        syscall_list = validate_syscall(syscall_list)
        predicts = TestSyzllmServer.fill_mask_for_test(syscall_list)
        for call in predicts:
            print(call)

    @staticmethod
    def fill_mask_for_test(syscall_list):
        input_ids_tensor = tokenizer.tokenize_sequence(syscall_list, return_tensors="pt", max_length_arg=max(128, highest_power_of_2(len(syscall_list) + 2) * 2))
        input_ids = input_ids_tensor.data['input_ids']
        mask_token_index = torch.where(input_ids == 196436)[1]
        mask_token_logits = mask_model(input_ids).logits[0, mask_token_index, :]

        top_tokens = TestSyzllmServer.topk_for_test(mask_token_logits)
        syscalls = []
        for token in top_tokens:
            call = tokenizer.decode([token])
            if "image" in call:
                continue
            syscalls.append(call)

        return syscalls

    @staticmethod
    def topk_for_test(logits, k=10):
        return torch.topk(logits, k, dim=1).indices[0].tolist()


if __name__ == '__main__':
    unittest.main()