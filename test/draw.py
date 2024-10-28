import datetime

import matplotlib.pyplot as plt
import re

syzLLM_ncalls_path = ['./syzllm-ncalls1.txt', './syzllm-ncalls2.txt', 'syzllm-0319-1.txt']
syzLLM_ncalls_broken_path = ['./syzllm-ncalls-broken-1.txt', 'syzllm-ncalls-broken-2.txt']
syzLLM_pure_path = ['./syzllm1.txt']
table_path = ['./table1.txt', './table2.txt', './table3.txt']

syzkaller = ['syzkaller-0319-1.txt', 'expt-log.txt', './temp/expt-log.txt']
syzllm = ['syzllm-0319-1.txt',
          'expt-syzllm-0320.txt',
          'expt-syzllm-0321.txt',
          'expt-syzllm-0321-2.txt',
          'expt-syzllm-0322-1.txt',
          'expt-syzllm-0322-2.txt',
          'expt-syzllm-0323-1.txt',
          'expt-syzllm-0324-1.txt',
          'expt-syzllm-0502-1.txt',
          'expt-syzllm-0724.txt',
          'expt-syzllm-0725.txt',
          'expt-syzllm-0726.txt',
          'expt-syzllm-0728.txt',
          'expt-distil-BS2-1e5.txt',
          'expt.txt',
          './temp/expt-syzllm-0324-1.txt',
          './temp/expt-res-sampling.txt',
          './temp/expt-res.txt',
          'expt-res.txt',
          'expt-res-sampling-0817.txt',
          'expt-syzllm-1028.txt',
          'expt-syzllm-batch.txt'
          ]


SyzLLM_label = 'SyzLLM'
SyzLLM_Batch_ResInline_label = 'SyzLLM-ResInline-Batch'
SyzLLM_ResInline_label = 'SyzLLM-ResInline'
syzkaller_label = 'Syzkaller'

color_map = {
    SyzLLM_label: 'r--',
    syzkaller_label: 'g--',
    SyzLLM_Batch_ResInline_label: 'b--',
    SyzLLM_ResInline_label: 'k-'
}


def calculate_time_differences(file_path):
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'cover' in line and 'executed' in line:
                time_match = re.search(r'\d{2}:\d{2}:\d{2}', line)
                if time_match:
                    times.append(time_match.group())

    time_diffs = [0.0]
    first_time = times[0]
    for time in times[1:]:
        time_diff = calculate_time_difference(first_time, time)
        time_diffs.append(time_diff)

    time_diffs = [time/3600 for time in time_diffs]
    return time_diffs


def calculate_time_difference(time1, time2):
    h1, m1, s1 = map(int, time1.split(':'))
    h2, m2, s2 = map(int, time2.split(':'))
    if h2 < h1:
        h2 += 24

    total_seconds1 = h1 * 3600 + m1 * 60 + s1
    total_seconds2 = h2 * 3600 + m2 * 60 + s2

    time_diff = total_seconds2 - total_seconds1

    return time_diff


def extract_coverage(file_path):
    covers = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'cover\s+(\d+),', line)
            if match:
                cover = int(match.group(1))
                covers.append(cover)
    return covers


def extract_execute(file_path):
    covers = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'executed\s+(\d+),', line)
            if match:
                cover = int(match.group(1))
                covers.append(cover)
    return covers


class Line(object):
    def __init__(self, file_path, label):
        self.X_time = calculate_time_differences(file_path)
        self.X_execute = extract_execute(file_path)
        self.Y = extract_coverage(file_path)
        self.label = label


def draw_lines_time(lines):
    for line in lines:
        plt.plot(line.X_time, line.Y, color_map[line.label], label=line.label)

    plt.xlabel('time /hours')
    plt.ylabel('coverage')
    plt.legend()
    plt.show()


def draw_lines_execute(lines):
    for line in lines:
        plt.plot(line.X_execute, line.Y, color_map[line.label], label=line.label)

    plt.xlabel('executed programs')
    plt.ylabel('coverage')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lines = [
        #Line(syzLLM_ncalls_path[2], SyzLLM_label),
        #Line(syzLLM_ncalls_path[1], SyzLLM_label),
        #Line(syzLLM_ncalls_broken_path[0], SyzLLM_broken_label),
        #Line(syzLLM_ncalls_broken_path[1], SyzLLM_broken_label),
        #Line(table_path[0], choiceTable_label),
        #Line(table_path[1], choiceTable_label),
        #Line(table_path[2], choiceTable_label),
        Line(syzkaller[1], syzkaller_label),
        Line(syzllm[7], SyzLLM_label),
        #Line(syzllm[8], SyzLLM_pure_label),
        #Line(syzllm[13], SyzLLM_broken_label),
        Line(syzllm[20], SyzLLM_ResInline_label),
        Line(syzllm[21], SyzLLM_Batch_ResInline_label)
        #Line(syzllm[6], SyzLLM_pure_label),
        #Line(syzllm[5], SyzLLM_broken_label),
        #Line(syzllm[2], SyzLLM_pure_label)
    ]

    # lines = [
    #     Line(syzkaller[2], syzkaller_label),
    #     Line(syzllm[15], SyzLLM_label),
    #     Line(syzllm[16], SyzLLM_pure_label),
    #     Line(syzllm[17], SyzLLM_broken_label)
    # ]

    draw_lines_execute(lines)
    draw_lines_time(lines)
