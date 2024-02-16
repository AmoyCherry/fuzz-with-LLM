import datetime

import matplotlib.pyplot as plt
import re

syzLLM_ncalls_path = ['./syzllm-ncalls1.txt', './syzllm-ncalls2.txt']
syzLLM_ncalls_broken_path = ['./syzllm-ncalls-broken-1.txt', 'syzllm-ncalls-broken-2.txt']
syzLLM_pure_path = ['./syzllm1.txt']
table_path = ['./table1.txt', './table2.txt', './table3.txt']


SyzLLM_label = 'SyzLLM'
SyzLLM_broken_label = 'SyzLLM-broken'
SyzLLM_pure_label = 'SyzLLM-pure'
choiceTable_label = 'ChoiceTable'

color_map = {
    SyzLLM_label : 'r--',
    choiceTable_label : 'g--',
    SyzLLM_broken_label : 'b--',
    SyzLLM_pure_label : 'rb-'
}


def calculate_time_differences(file_path):
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'cover' in line and 'executed' in line:
                time_match = re.search(r'\d{2}:\d{2}:\d{2}', line)
                if time_match:
                    times.append(time_match.group())

    time_diffs = [0]
    first_time = times[0]
    for time in times[1:]:
        time_diff = calculate_time_difference(first_time, time)
        time_diffs.append(time_diff)

    return time_diffs


def calculate_time_difference(time1, time2):
    h1, m1, s1 = map(int, time1.split(':'))
    h2, m2, s2 = map(int, time2.split(':'))

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

    plt.xlabel('time /sec')
    plt.ylabel('coverage')
    plt.legend()
    plt.show()


def draw_lines_execute(lines):
    for line in lines:
        plt.plot(line.X_execute, line.Y, color_map[line.label], label=line.label)

    plt.xlabel('executed calls')
    plt.ylabel('coverage')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lines = [
        Line(syzLLM_ncalls_path[0], SyzLLM_label),
        Line(syzLLM_ncalls_path[1], SyzLLM_label),
        Line(syzLLM_ncalls_broken_path[0], SyzLLM_broken_label),
        Line(syzLLM_ncalls_broken_path[1], SyzLLM_broken_label),
        #Line(table_path[0], choiceTable_label),
        #Line(table_path[1], choiceTable_label),
        Line(table_path[2], choiceTable_label),
    ]

    #draw_lines_execute(lines)
    draw_lines_time(lines)
