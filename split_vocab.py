import argparse


def split_file(input_file, output_prefix, programs_per_file):
    with open(input_file, 'r') as f:
        current_file = 1
        current_program_count = 0
        current_program = []
        output = open(f"{output_prefix}_{current_file}.txt", 'w')

        excluded_tokens = ['[UNK]', 'MASK', 'CLS', 'PAD']
        for line in f:
            if any(token in line for token in excluded_tokens):
                continue
            current_program.append(line.strip())
            if line.strip() == "[SEP]":
                output.write('\n'.join(current_program) + '\n')
                current_program = []
                current_program_count += 1

                if current_program_count == programs_per_file:
                    output.close()
                    current_file += 1
                    current_program_count = 0
                    output = open(f"{output_prefix}_{current_file}.txt", 'w')

        # Write any remaining lines
        if current_program:
            output.write('\n'.join(current_program) + '\n')
        output.close()


def main():
    input_file = "./DummySyzTokenizer/vocab.txt"
    output_prefix = "./tokens/"
    programs_per_file = 512
    split_file(input_file, output_prefix, programs_per_file)


if __name__ == "__main__":
    main()