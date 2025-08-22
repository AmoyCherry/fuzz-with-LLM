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
          'expt-syzllm-diverse-22M-0513.txt',
          'eval-syzkaller.txt',
          'eval-syzllm.txt',
          'eval-syzkaller-noreproduce-40h.txt',
          'eval-syzllm-no-reproduce.txt',
          'logs/ripple-syzkaller.txt', # 25
          'logs/ripple-syzllm.txt', # 26
          'logs/ripple-syzkaller-48h.txt', # 27
          'logs/ripple-syzllm-48h.txt', # 28
          'logs/logs-vanilla-syzkaller-0810.txt', # 29
          'logs/logs-syzllm-0810.txt', # 30
          'logs/logs-syzkaller-no-seeds.txt', # 31
          'logs/logs-syzllm-no-seeds.txt', # 32
          'logs/logs-optimizer.txt', # 33
          'logs/logs-optimizer-reduce-09-05-2h-001.txt', # 34
          'logs/logs-optimizer-reduce-09-08-3h-025.txt', # 35
          'logs/logs-optimizer-03.txt', # 36
          ]


SyzLLM_label = 'SyzLLM-no-opt'
SyzLLM_optimizer_reduce_label = 'SyzLLM'
SyzLLM_optimizer_label = 'SyzLLM-optimizer'
syzkaller_label = 'Syzkaller'
diverse = 'SyzLLM-Diverse-22M'

color_map = {
    SyzLLM_label: 'b-.',
    syzkaller_label: 'g--',
    SyzLLM_optimizer_reduce_label: 'r',
    SyzLLM_optimizer_label: 'k-',
    diverse: 'c-',
}


def calculate_time_differences(file_path):
    times = []
    days = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'coverage' in line and 'candidates' in line:
                time_match = re.search(r'\d{2}:\d{2}:\d{2}', line)
                day_match = re.search(r'\d{4}/\d{2}/\d{2}', line)
                if time_match and day_match:
                    times.append(time_match.group())
                    days.append(day_match.group())

    time_diffs = [0.0]
    first_time = times[0]
    for i in range(1, len(times)):
        time_diff = calculate_time_difference(first_time, days[0], times[i], days[i])
        time_diffs.append(time_diff)

    time_diffs = [time/3600 for time in time_diffs]
    return time_diffs


def calculate_time_difference(time1, first_day, time2, time2_day):
    y1, mon1, d1 = map(int, first_day.split('/'))
    h1, min1, s1 = map(int, time1.split(':'))
    dt1 = datetime.datetime(y1, mon1, d1, h1, min1, s1)

    y2, mon2, d2 = map(int, time2_day.split('/'))
    h2, min2, s2 = map(int, time2.split(':'))
    dt2 = datetime.datetime(y2, mon2, d2, h2, min2, s2)

    diff = dt2 - dt1
    return diff.total_seconds()


def extract_coverage(file_path):
    covers = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r' coverage=(\d+) ', line)
            if match:
                cover = int(match.group(1))
                covers.append(cover)
    return covers


def extract_execute(file_path):
    covers = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r' total=(\d+) ', line)
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


def draw_lines_time(lines, save_path='coverage_vs_time.pdf'):
    fig, ax = plt.subplots()
    for line in lines:
        ax.plot(line.X_time, line.Y, color_map[line.label], label=line.label)

    ax.set_xlabel('time(hours)', fontsize=13)
    ax.set_ylabel('branches coverage', fontsize=13)
    ax.legend(loc='lower right', fontsize=16)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory


def draw_lines_execute(lines, save_path='coverage_vs_executions.pdf'):
    fig, ax = plt.subplots()
    for line in lines:
        ax.plot(line.X_execute, line.Y, color_map[line.label], label=line.label)

    ax.set_xlabel('executed programs', fontsize=13)
    ax.set_ylabel('branches coverage', fontsize=13)
    ax.legend(loc='lower right', fontsize=16)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory


if __name__ == '__main__':
    lines = [
        #Line(syzLLM_ncalls_path[2], SyzLLM_label),
        #Line(syzLLM_ncalls_path[1], SyzLLM_label),
        #Line(syzLLM_ncalls_broken_path[0], SyzLLM_broken_label),
        #Line(syzLLM_ncalls_broken_path[1], SyzLLM_broken_label),
        #Line(table_path[0], choiceTable_label),
        #Line(table_path[1], choiceTable_label),
        #Line(table_path[2], choiceTable_label),
        #Line(syzkaller[1], syzkaller_label),
        #Line(syzllm[23], syzkaller_label),
        #Line(syzllm[8], SyzLLM_pure_label),
        #Line(syzllm[13], SyzLLM_broken_label),
        #Line(syzllm[18], SyzLLM_broken_label),
        #Line(syzllm[19], SyzLLM_pure_label),
        Line(syzllm[27], syzkaller_label),
        #Line(syzllm[28], SyzLLM_label),
        #Line(syzllm[35], SyzLLM_optimizer_label),
        Line(syzllm[36], SyzLLM_optimizer_reduce_label),
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